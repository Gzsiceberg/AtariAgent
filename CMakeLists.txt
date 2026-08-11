cmake_minimum_required(VERSION 3.20)
project(atariagent_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_mcts_native MODULE cpp/mcts_native.cpp)
target_compile_features(_mcts_native PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(_mcts_native PRIVATE /O2)
else()
    target_compile_options(_mcts_native PRIVATE -O3)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(_mcts_native PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _mcts_native DESTINATION atariagent/search)
