source emsdk/emsdk_env.sh

emcmake cmake .
emmake make -j 

mv cartographer_exec.js ../cartographer-webserver/wasm/cartographer_exec.js
mv cartographer_exec.wasm ../cartographer-webserver/wasm/cartographer_exec.wasm