cmake_minimum_required(VERSION 3.20)
project(atariagent_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)
find_package(OpenMP)

function(configure_torch_extension target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(
        ${target}
        PRIVATE
            ${TORCH_LIBRARIES}
            ${TORCH_PYTHON_LIBRARY}
    )
    set_target_properties(
        ${target}
        PROPERTIES
            BUILD_RPATH "${TORCH_LIBRARY_DIR}"
            INSTALL_RPATH "$ORIGIN/../../torch/lib"
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2)
    else()
        target_compile_options(${target} PRIVATE -O3)
    endif()
endfunction()

pybind11_add_module(
    _mcts_native
    MODULE
        cpp/mcts_bindings.cpp
        cpp/native_search.cpp
        cpp/search/batch_tree.cpp
        cpp/reanalysis/engine.cpp
)
target_include_directories(_mcts_native PRIVATE cpp)
configure_torch_extension(_mcts_native)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_mcts_native PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_mcts_native PRIVATE ATARIAGENT_HAS_OPENMP=1)
endif()

pybind11_add_module(
    _models_native
    MODULE
        cpp/models_bindings.cpp
        cpp/native_search.cpp
        cpp/models/state_dict.cpp
        cpp/models/representation.cpp
        cpp/models/dynamics.cpp
        cpp/models/prediction.cpp
        cpp/inference/evaluator.cpp
        cpp/inference/mcts.cpp
        cpp/inference/value_target.cpp
)
target_include_directories(_models_native PRIVATE cpp)
configure_torch_extension(_models_native)

install(TARGETS _mcts_native DESTINATION atariagent/search)
install(TARGETS _models_native DESTINATION atariagent/models)
