import { useState } from 'react'
import * as tf from "@tensorflow/tfjs"
import './App.css'
import { useEffect } from 'react'
import cartographer_module from "../wasm/cartographer_exec.js"

async function LoadModel(model_url) { 
  const model = await tf.loadLayersModel(model_url)
  console.log(model)
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
    const map_data = new Uint8Array(reader.result);

    cartographer_module().then((Module) => { 
        Module['FS_createDataFile']('/', fileName, map_data, true, true, true)
        var backend = new Module.CartographerBackend(fileName, "FWT", "PHWT")

        let prediction_data = backend.generate_prediction_data()
        console.log(prediction_data)
        // backend.delete()
        
        // let translation_list_binding = prediction_data.translation_list
        // let translation_list = []

        // for (let i =0 ; i < translation_list_binding.size(); i++) { 
        //   let current_i = translation_list_binding.get(i)
        //   translation_list.push([current_i.get(0), current_i.get(1), current_i.get(2)])
        // }

        // let interpolated_grid_binding = prediction_data.interpolated_grid
        // let interpolated_grid = []

        // for (let i = 0; i < interpolated_grid_binding.size(); i++) 




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
