import { useState } from 'react'
import * as tf from "@tensorflow/tfjs"
import './App.css'
import { useEffect } from 'react'
import cartographer_module from "../wasm/cartographer_exec.js"

async function LoadModel(model_url) { 
  const model = await tf.loadLayersModel(model_url)
  return model
}

// async function LoadPython() { 
  
//   let pyodide = await window.loadPyodide();
//   await pyodide.loadPackage("micropip");
//   await pyodide.loadPackage("numpy");

//   console.log(
//     pyodide.runPython(
//       `
//       import numpy as np
//       print(np.zeros((3,3,3)))
//       `
//     )
//   )
// }

function App() {

  useEffect(() => {
    // LoadModel("js_model/model.json")
    // LoadPython()
  })

  let fileName = ""
  let reader = new FileReader()

  function LoadBackend() { 
    LoadModel("model_no_bn/model.json").then((model) => {

      console.log(model)

    const map_data = new Uint8Array(reader.result);

    cartographer_module().then((Module) => { 
        Module['FS_createDataFile']('/', fileName, map_data, true, true, true)
        var backend = new Module.CartographerBackend(fileName, "FWT", "PHWT")

        let prediction_data = backend.generate_prediction_data()
        // backend.delete()
        
        let translation_list_binding = prediction_data.translation_list
        let translation_list = []

        for (let i =0 ; i < translation_list_binding.size(); i++) { 
          let current_i = translation_list_binding.get(i)
          translation_list.push([current_i.get(0), current_i.get(1), current_i.get(2)])
        }

        let interpolated_grid_binding = prediction_data.interpolated_grid

        let interpolated_map = tf.zeros([prediction_data.num_x, prediction_data.num_y, prediction_data.num_z])

        interpolated_map.buffer().then((buffer) => {
          console.log(buffer)
          for (let i = 0; i < interpolated_grid_binding.size(); i++) {
            let x_data = interpolated_grid_binding.get(i);

            for (let j = 0; j < x_data.size(); j++) {
              let y_data = x_data.get(j)

              for (let k = 0; k < y_data.size(); k++) {
                let z_data = y_data.get(k)
                buffer.set(z_data, i,j,k)
              }
              }
            }
          })
        
        // let interpolated_tensor = tf.tensor3d(interpolated_grid, prediction_data.num_x, prediction_data.num_y, prediction_data.num_z])
        let overlap = 16

        let predicted_map = tf.zeros([
          32*prediction_data.na + (32-overlap),
          32*prediction_data.nb + (32-overlap),
          32*prediction_data.nc + (32-overlap)
        ])

        let count_map = tf.zeros([
          32*prediction_data.na + (32-overlap),
          32*prediction_data.nb + (32-overlap),
          32*prediction_data.nc + (32-overlap)
        ])
        
        for (let i = 0; i < translation_list.length; i++) {
          let x = translation_list[i][0]
          let y = translation_list[i][1]
          let z = translation_list[i][2]

          let subarray = interpolated_map.slice([x,y,z], [32,32,32]).reshape([1, 32,32,32,1])

          let predicted_sub = model.predict(subarray).squeeze()
          tf.argMax(predicted_sub).buffer((pred_buffer) => { 
            for (let a = x, f = 0; a < x+32; a++, f++) { 
              for (let b = y, g = 0; b < y+32; b++, g++) { 
                for (let c = z, h = 0; c < z+32; c++, h++) { 
                  console.log(a,b,c,f,g,h, argmax)
                  predicted_map[a][b][c] += pred_buffer[f][g][h]
                  count_map[a][b][c] += 1
                }
              }
            }
          })
        }

        let sliced_pred_map = predicted_map.slice([0,0,0], [32*prediction_data.na, 32*prediction_data.nb, 32*prediction_data.nc])
        let sliced_count_map = count_map.slice([0,0,0], [32*prediction_data.na, 32*prediction_data.nb, 32*prediction_data.nc])

        let final_map = sliced_pred_map.div(sliced_count_map)

        // NOW NEED TO SEND THIS BACK TO C++ TO GET MADE INTO A MAP WHICH CAN BE WRITTEN AND READ BY MOORHEN!


      })
    })
  }

  const load_file = (file) => { 
    fileName = file.name
    reader.addEventListener('loadend', LoadBackend);
    reader.readAsArrayBuffer(file);
}

  return (
    <>
      <input type="file" onChange={(e) => load_file(e.target.files[0])}/>
    </>
  )
}

export default App
