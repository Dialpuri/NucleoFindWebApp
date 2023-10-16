#include <emscripten.h>
#include <emscripten/bind.h>

#include "gemmi/mtz.hpp"
#include "gemmi/fourier.hpp"
#include "gemmi/ccp4.hpp"     
#include "gemmi/asumask.hpp"     

#include <iostream>
#include <algorithm>

struct CartographerPrePredictionData { 
    std::vector<std::vector<std::vector<float>>> interpolated_grid;
    std::vector<std::vector<int>> translation_list; 
    float na;
    float nb; 
    float nc; 
    int num_x, num_y, num_z;
};

class CartographerBackend { 
    public: 
        CartographerBackend(const std::string& path, const std::string& f, const std::string& phi) : mtz_path(path), f_label(f), phi_label(phi) {}

        void load_mtz_file() { 
            std::cout << "Loading MTZ File" << std::endl;
            gemmi::Mtz mtz = gemmi::read_mtz_file(mtz_path);

            const gemmi::Mtz::Column& f = mtz.get_column_with_label(f_label);
            const gemmi::Mtz::Column& phi = mtz.get_column_with_label(phi_label);

            gemmi::FPhiProxy<gemmi::MtzDataProxy> fphi (gemmi::MtzDataProxy{mtz}, f.idx, phi.idx);

            grid = gemmi::transform_f_phi_to_map2<float>(fphi, {0,0,0}, 0.0, {0,0,0});
        }

        gemmi::Box<gemmi::Position> get_bounding_box() { 
            gemmi::Box<gemmi::Fractional> extent = gemmi::find_asu_brick(grid.spacegroup).get_extent();
            extent.maximum = gemmi::Fractional({1,1,1});
            extent.minimum = gemmi::Fractional({0,0,0});

            std::vector<gemmi::Fractional> fractional_corners = {extent.minimum,
                gemmi::Fractional({extent.maximum.x, extent.minimum.y, extent.minimum.z}),
                gemmi::Fractional({extent.minimum.x, extent.maximum.y, extent.minimum.z}),
                gemmi::Fractional({extent.minimum.x, extent.minimum.y, extent.maximum.z}),
                gemmi::Fractional({extent.maximum.x, extent.maximum.y, extent.minimum.z}),
                gemmi::Fractional({extent.maximum.x, extent.minimum.y, extent.maximum.z}),
                gemmi::Fractional({extent.minimum.x, extent.maximum.y, extent.maximum.z}),
                extent.maximum
            };

            std::vector<gemmi::Position> orthogonal_corners; 

            for (const auto& corner: fractional_corners) {
                orthogonal_corners.emplace_back(grid.unit_cell.orthogonalize(corner));
            }

            std::vector<double> x_corners; 
            std::vector<double> y_corners; 
            std::vector<double> z_corners; 

            for (const auto& corner: orthogonal_corners) {
                x_corners.push_back(corner.x);
                y_corners.push_back(corner.y);
                z_corners.push_back(corner.z);
            }

            auto min_x = (*std::min_element(x_corners.begin(), x_corners.end()));
            auto min_y = (*std::min_element(y_corners.begin(), y_corners.end()));
            auto min_z = (*std::min_element(z_corners.begin(), z_corners.end()));

            auto max_x = (*std::max_element(x_corners.begin(), x_corners.end()));
            auto max_y = (*std::max_element(y_corners.begin(), y_corners.end()));
            auto max_z = (*std::max_element(z_corners.begin(), z_corners.end()));

            gemmi::Box<gemmi::Position> box; 
            box.maximum = gemmi::Position(gemmi::Vec3(max_x, max_y, max_z));
            box.minimum =  gemmi::Position(gemmi::Vec3(min_x, min_y, min_z));
            return box;
        }

        void interpolate_grid(float grid_spacing = 0.7) { 
            gemmi::Box<gemmi::Position> box = get_bounding_box();
            gemmi::Position size = box.get_size();

            num_x = round(size.x / grid_spacing);
            num_y = round(size.y / grid_spacing);
            num_z = round(size.z / grid_spacing);

            gemmi::Mat33 scale = {
                grid_spacing, 0, 0,
                0, grid_spacing, 0,
                0, 0, grid_spacing            
            };
            
            gemmi::Transform tr = {scale, box.minimum};

            std::vector<std::vector<std::vector<float>>> array(num_x, std::vector<std::vector<float>>(num_y, std::vector<float>(num_z, 0.0)));

            for (int i = 0; i < array.size(); ++i) {
                for (int j = 0; j < array[0].size(); ++j) {
                    for (int k = 0; k < array[0][0].size(); ++k) {
                        gemmi::Position pos(tr.apply(gemmi::Vec3(i, j, k)));
                        gemmi::Fractional fpos = grid.unit_cell.fractionalize(pos);
                        array[i][j][k] = grid.interpolate(fpos, 2);
                    }
                }
            }

            interpolated_grid = array; 

        }


        void calculate_translation(int overlap = 32) { 
            float overlap_na = ceil(interpolated_grid.size() / overlap);
            float overlap_nb = ceil(interpolated_grid[0].size() / overlap);
            float overlap_nc = ceil(interpolated_grid[0][0].size() / overlap);


            for (int i = 0; i < overlap_na; i++) { 
                for (int j = 0; j < overlap_nb; j++) { 
                    for (int k = 0; k < overlap_nc; k++) { 
                        translation_list.push_back({i*overlap, j*overlap, k*overlap});
                    }
                }
            }

            na = ceil(interpolated_grid.size() / 32);
            nb = ceil(interpolated_grid[0].size() / 32);
            nc = ceil(interpolated_grid[0][0].size() / 32);
        }

        CartographerPrePredictionData generate_prediction_list() {
            load_mtz_file();
            interpolate_grid();
            calculate_translation(); 

            CartographerPrePredictionData data; 
            data.interpolated_grid = interpolated_grid; 
            data.translation_list = translation_list;
            data.na = na;
            data.nb = nb; 
            data.nc = nc; 
            data.num_x = num_x;
            data.num_y = num_y; 
            data.num_z = num_z; 

            return data;
        }



    private:
        std::string mtz_path; 
        std::string f_label;
        std::string phi_label;

        gemmi::Grid<float> grid; 
        std::vector<std::vector<std::vector<float>>> interpolated_grid; 
        std::vector<std::vector<int>> translation_list; 
        float na;
        float nb;
        float nc; 
        int num_x; 
        int num_y; 
        int num_z; 

};

// extern "C" void load_mtz_file(const std::string& path) { 
//     gemmi::Mtz mtz = gemmi::read_mtz_file(path);

//     const gemmi::Mtz::Column& f = mtz.get_column_with_label("FWT");
//     const gemmi::Mtz::Column& phi = mtz.get_column_with_label("PHWT");

//     gemmi::FPhiProxy<gemmi::MtzDataProxy> fphi (gemmi::MtzDataProxy{mtz}, f.idx, phi.idx);

//     gemmi::Grid<float> grid = gemmi::transform_f_phi_to_map2<float>(
//         fphi, {0,0,0}, 0.0, {0,0,0}
//     );

//     std::cout << "Calculated grid = " << grid.nu << " " << grid.nv << " " << grid.nw << std::endl;


// }


using namespace emscripten;

EMSCRIPTEN_BINDINGS(cartographer_module) { 
    // function("load_mtz_file", &load_mtz_file);
    register_vector<int>("VectorOfInts");
    register_vector<std::vector<int>>("VectorVectorOfInts");

    register_vector<float>("VectorOfFloats");
    register_vector<std::vector<float>>("VectorVectorOfFloats");
    register_vector<std::vector<std::vector<float>>>("VectorVectorVectorOfFloats");

    value_object<CartographerPrePredictionData>("CargorapherPrePredictionData")
        .field("interpolated_grid", &CartographerPrePredictionData::interpolated_grid)
        .field("translation_list", &CartographerPrePredictionData::translation_list)
        .field("na", &CartographerPrePredictionData::na)
        .field("nb", &CartographerPrePredictionData::nb)
        .field("nc", &CartographerPrePredictionData::nc)
        .field("num_x", &CartographerPrePredictionData::num_x)
        .field("num_y", &CartographerPrePredictionData::num_y)
        .field("num_z", &CartographerPrePredictionData::num_z);

    class_<CartographerBackend>("CartographerBackend")
        .constructor<const std::string&, const std::string&, const std::string&>()
        .function("generate_prediction_data", &CartographerBackend::generate_prediction_list);

}