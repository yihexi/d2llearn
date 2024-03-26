cmake_minimum_required(VERSION 3.25)
project(LinearConvert)

set(CMAKE_CXX_STANDARD 14)


# tensorrt
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(TensorRT REQUIRED)
MESSAGE(STATUS "TensorRT Version:           ${TENSORRT_VERSION}")
MESSAGE(STATUS "TensorRT Root(optional):    ${TENSORRT_ROOT_DIR}")
message(STATUS "TensorRT Include Dir:       ${TENSORRT_INCLUDE}")
message(STATUS "TensorRT Library Dir:       ${TENSORRT_LIBRARY_LIB}")

foreach (LIB ${TENSORRT_LIBRARY_LIB})
    message(STATUS "TensorRT Library:           ${LIB}")
endforeach ()


add_executable(LinearConvert main.cpp)