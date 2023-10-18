#include <emscripten.h>
#include <emscripten/bind.h>

#include "gemmi/mtz.hpp"
#include "gemmi/fourier.hpp"
#include "gemmi/ccp4.hpp"     
#include "gemmi/asumask.hpp"     
#include "gemmi/model.hpp"

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
            grid.normalize();
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
            std::cout << "Interpolated grid size " << interpolated_grid.size() << " " << interpolated_grid[0].size() << " " << interpolated_grid[0][0].size() << std::endl; 

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

            na = floor(interpolated_grid.size() / 32) + 1 ;
            nb = floor(interpolated_grid[0].size() / 32) + 1;
            nc = floor(interpolated_grid[0][0].size() / 32) + 1;
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

        void reinterpret_to_output(std::vector<std::vector<std::vector<float>>>& array) { 
            std::cout << "Reinterpreting to output" << std::endl;
            std::cout << na << " " << nb << " " << nc << std::endl;
            gemmi::Structure s;
            s.cell = grid.unit_cell;
            s.spacegroup_hm = grid.spacegroup->hm;

            std::cout << "Input Array Shape " << array.size() << ", " << array[0].size() << ", " << array[0][0].size() << std::endl; 

            std::cout << "Input Grid Unit Cell " << grid.unit_cell.a << ", " << grid.unit_cell.b << ", " << grid.unit_cell.c << std::endl; 


            gemmi::Grid<float> output_grid;
            output_grid.spacegroup = grid.spacegroup;
            output_grid.set_unit_cell(grid.unit_cell);
            output_grid.set_size_from_spacing(0.7, gemmi::GridSizeRounding::Nearest);

            std::cout << "Output Grid Unit Cell " << output_grid.unit_cell.a << ", " << output_grid.unit_cell.b << ", " << output_grid.unit_cell.c << std::endl; 
            std::cout << "Output Grid N " << output_grid.nu << ", " << output_grid.nv << ", " << output_grid.nw << std::endl; 


            float size_x = 0.7 * array.size();
            float size_y = 0.7 * array[0].size();
            float size_z = 0.7 * array[0][0].size();

            gemmi::UnitCell array_cell(size_x, size_y, size_z, 90, 90, 90);

            std::cout << "Array Cell Unit Cell " << array_cell.a << ", " << array_cell.b << ", " << array_cell.c << std::endl; 

            gemmi::Grid array_grid;
            array_grid.set_size(array.size(), array[0].size(), array[0][0].size());
            array_grid.unit_cell = array_cell;


            for (int i = 0; i < array.size(); i++) {
                for (int j = 0; j < array[0].size(); j++) {
                    for (int k = 0; k < array[0][0].size(); k++) {
                        array_grid.data[grid.index_s(i,j,k)] = array[i][j][k];
                    }
                }
            }

            
            // array_grid.unit_cell = grid.unit_cell;
            // array_grid.spacegroup = grid.spacegroup;

            std::cout << "Beginning masked asu interpolation" << std::endl;


            auto masked_asu = gemmi::masked_asu(output_grid);

            for (auto it = masked_asu.begin(); it != masked_asu.end(); ++it) {
                gemmi::Position position = output_grid.point_to_position(*it) - box.minimum;
                *(*it).value = array_grid.interpolate_value(position);
            }

            output_grid.symmetrize_max();

            gemmi::Ccp4<float> map; 
            // map.grid = *(masked_asu.grid); 
            map.grid = output_grid;

            map.update_ccp4_header();
            map.write_ccp4_map("/predicted.map");
            std::cout << "Written CCP4 Map" << std::endl;

        }



    private:
        std::string mtz_path; 
        std::string f_label;
        std::string phi_label;

        gemmi::Box<gemmi::Position> box; 
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
        .function("generate_prediction_data", &CartographerBackend::generate_prediction_list)
        .function("reinterpret_to_output", &CartographerBackend::reinterpret_to_output);

}