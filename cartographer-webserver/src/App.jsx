import { useState, useRef } from 'react'
import * as tf from "@tensorflow/tfjs"
import './App.css'
import { useEffect } from 'react'
import cartographer_module from "../wasm/cartographer_exec.js"
import { MoorhenContainer, MoorhenContextProvider, MoorhenMap, MoorhenMolecule } from 'moorhen'

async function LoadModel(model_url) {
  const model = await tf.loadLayersModel(model_url)
  return model
}


async function calculate_predicted_map(backend, model) {
  let prediction_data = backend.generate_prediction_data()
  // backend.delete()

  let translation_list_binding = prediction_data.translation_list
  let translation_list = []

  for (let i = 0; i < translation_list_binding.size(); i++) {
    let current_i = translation_list_binding.get(i)
    translation_list.push([current_i.get(0), current_i.get(1), current_i.get(2)])
  }

  let interpolated_grid_binding = prediction_data.interpolated_grid

  let interpolated_map = tf.zeros([prediction_data.num_x, prediction_data.num_y, prediction_data.num_z])

  let interpolated_map_buffer = await interpolated_map.buffer()

  for (let i = 0; i < interpolated_grid_binding.size(); i++) {
    let x_data = interpolated_grid_binding.get(i);

    for (let j = 0; j < x_data.size(); j++) {
      let y_data = x_data.get(j)

      for (let k = 0; k < y_data.size(); k++) {
        let z_data = y_data.get(k)
        interpolated_map_buffer.set(z_data, i, j, k)
      }
    }
  }

  // let interpolated_tensor = tf.tensor3d(interpolated_grid, prediction_data.num_x, prediction_data.num_y, prediction_data.num_z])
  let overlap = 16

  let predicted_map = tf.zeros([
    32 * prediction_data.na + (32 - overlap),
    32 * prediction_data.nb + (32 - overlap),
    32 * prediction_data.nc + (32 - overlap)
  ])

  let count_map = tf.zeros([
    32 * prediction_data.na + (32 - overlap),
    32 * prediction_data.nb + (32 - overlap),
    32 * prediction_data.nc + (32 - overlap)
  ])

  let predicted_map_buffer = await predicted_map.buffer()
  let count_map_buffer = await count_map.buffer()

  for (let i = 0; i < translation_list.length; i++) {
    let x = translation_list[i][0]
    let y = translation_list[i][1]
    let z = translation_list[i][2]

    let subarray = interpolated_map.slice([x, y, z], [32, 32, 32]).reshape([1, 32, 32, 32, 1])

    let predicted_sub = model.predict(subarray).reshape([32, 32, 32, 2])


    let pred_buffer = await tf.argMax(predicted_sub, -1).buffer()

    for (let a = x, f = 0; a < x + 32; a++, f++) {
      for (let b = y, g = 0; b < y + 32; b++, g++) {
        for (let c = z, h = 0; c < z + 32; c++, h++) {
          let new_map_buffer_value = predicted_map_buffer.get(a, b, c) + pred_buffer.get(f, g, h)
          let new_count_buffer_value = count_map_buffer.get(a, b, c) + 1

          predicted_map_buffer.set(new_map_buffer_value, a, b, c)
          count_map_buffer.set(new_count_buffer_value, a, b, c)
        }
      }
    }

  }

  let sliced_pred_map = predicted_map.slice([0, 0, 0], [32 * prediction_data.na, 32 * prediction_data.nb, 32 * prediction_data.nc])
  let sliced_count_map = count_map.slice([0, 0, 0], [32 * prediction_data.na, 32 * prediction_data.nb, 32 * prediction_data.nc])

  let final_map = sliced_pred_map.div(sliced_count_map)

  // NOW NEED TO SEND THIS BACK TO C++ TO GET MADE INTO A MAP WHICH CAN BE WRITTEN AND READ BY MOORHEN!
  let final_map_data = await final_map.array()

  // final_map = final_map.reshape( [32 * prediction_data.na, 32 * prediction_data.nb, 32 * prediction_data.nc, 1])
  // const {values, indices} = tf.unique(final_map, -1)
  // values.print()
  // indices.print()
  return final_map_data

}


function App() {


  let fileName = ""
  let reader = new FileReader()
  const [cootInitialized, setCootInitialized] = useState(false)
  const controls = useRef();

  async function LoadBackend() {
    const map_data = new Uint8Array(reader.result);
    display_map(map_data)

    LoadModel("js_model/model.json").then((model) => {
      // console.log(model)
      const map_data = new Uint8Array(reader.result);

      cartographer_module().then((Module) => {
        Module['FS_createDataFile']('/', fileName, map_data, true, true, true)
        var backend = new Module.CartographerBackend(fileName, "FWT", "PHWT")
        calculate_predicted_map(backend, model).then((predicted_map) => {
          console.log("JS Finished processing pred map, sending to C++")
          let predicted_map_array = Array.from(predicted_map)

          var return_vector3d = new Module["VectorVectorVectorOfFloats"]()

          for (let i = 0; i < predicted_map.length; i++) {
            var return_vector2d = new Module["VectorVectorOfFloats"]()
            for (let j = 0; j < predicted_map[0].length; j++) {
              var return_vector1d = new Module["VectorOfFloats"]()
              for (let k = 0; k < predicted_map[0][0].length; k++) {
                return_vector1d.push_back(predicted_map_array[i][j][k])
              }
              return_vector2d.push_back(return_vector1d)
            }
            return_vector3d.push_back(return_vector2d)
          }

          backend.reinterpret_to_output(return_vector3d)


          let map_file = Module["FS_readFile"]("/predicted.map")
          // console.log(map_file)

          const downloadURL = (data, fileName) => {
            const a = document.createElement('a')
            a.href = data
            a.download = fileName
            document.body.appendChild(a)
            a.style.display = 'none'
            a.click()
            a.remove()
          }
          
          const downloadBlob = (data, fileName, mimeType) => {
          
            const blob = new Blob([data], {
              type: mimeType
            })
          
            const url = window.URL.createObjectURL(blob)
          
            downloadURL(url, fileName)
          
            setTimeout(() => window.URL.revokeObjectURL(url), 1000)
          }

          downloadBlob(map_file, "predicted_map.map", "application/octet-stream")

          const map = new MoorhenMap(controls.current.commandCentre, controls.current.glRef);
          
            map.loadToCootFromMapData(map_file, "map-2", false);
            controls.current.changeMaps({ action: "Add", item: map })
            controls.current.setActiveMap(map)

        })

      })
    })
  }

  const load_file = (file) => {
    fileName = file.name
    reader.addEventListener('loadend', LoadBackend);
    reader.readAsArrayBuffer(file);


  }

  const forwardControls = (forwardedControls) => {
    setCootInitialized(true)
    controls.current = forwardedControls
  }

  const display_map = (map_data) => {
    if (cootInitialized) {
      const map = new MoorhenMap(controls.current.commandCentre, controls.current.glRef);
      const mapMetadata = {
        F: "FWT",
        PHI: "PHWT",
        Fobs: "FP",
        SigFobs: "SIGFP",
        FreeR: "FREE",
        isDifference: false,
        useWeight: false,
        calcStructFact: true,
      }
      map.loadToCootFromMtzData(map_data, "map-name", mapMetadata);
      controls.current.changeMaps({ action: "Add", item: map })
      controls.current.setActiveMap(map)
    }
  }

  return (
    <>
      <input type="file" onChange={(e) => load_file(e.target.files[0])} />
      <MoorhenContextProvider defaultBackgroundColor={[51, 65, 85, 1]}>
        <MoorhenContainer forwardControls={forwardControls} setMoorhenDimensions={() => {
          return [1200, 800];
        }} viewOnly={false} />

      </MoorhenContextProvider>
    </>
  )
}

export default App
