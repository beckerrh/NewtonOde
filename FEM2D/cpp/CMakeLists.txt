cmake_minimum_required(VERSION 3.18)

project(FEM2D_mesh_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_meshcpp bindings.cpp refine_nvb.cpp topology.cpp geometry.cpp mesh_rebuild.cpp)

target_compile_features(_meshcpp PRIVATE cxx_std_17)

get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

set_target_properties(_meshcpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT}/python/FEM2D"
)