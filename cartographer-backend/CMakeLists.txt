cmake_minimum_required(VERSION 3.13)
project(Cartographer)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wno-enum-constexpr-conversion -I${CMAKE_INSTALL_PREFIX}/include -s USE_PTHREADS=1 -pthread")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(gemmi)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    checkout/gemmi/include)
    
add_executable(cartographer_exec ${CMAKE_CURRENT_SOURCE_DIR}/src/cartographer-backend.cpp)
set_target_properties (cartographer_exec PROPERTIES OUTPUT_NAME cartographer_exec )
target_link_options(cartographer_exec PRIVATE -sDEMANGLE_SUPPORT=1 -sASSERTIONS=2 -fexceptions -sALLOW_MEMORY_GROWTH=1 --bind -sFORCE_FILESYSTEM=1 -sMODULARIZE=1 -sEXPORT_NAME=cartographer_module -sEXPORTED_RUNTIME_METHODS=['FS'] -sMODULARIZE=1 -sEXPORT_ES6=1 --bind -lnodefs.js --profiling)

target_link_directories(cartographer_exec PUBLIC ../checkout/gemmi)
set_target_properties(cartographer_exec PROPERTIES COMPILE_FLAGS "-s ENVIRONMENT='web'")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cartographer.wasm
${CMAKE_CURRENT_BINARY_DIR}/cartographer.js
DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../cartographer-webserver/wasm)