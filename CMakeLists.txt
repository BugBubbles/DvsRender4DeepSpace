cmake_minimum_required(VERSION 3.22)
project(dvs_simulator_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 QUIET CONFIG)
if(NOT TARGET Eigen3::Eigen)
    file(GLOB EIGEN_CONDA_PKG_DIRS "$ENV{HOME}/miniconda3/pkgs/eigen-*/include/eigen3")
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
        HINTS
            $ENV{CONDA_PREFIX}/include/eigen3
            /usr/include/eigen3
            /usr/local/include/eigen3
        PATHS
            ${EIGEN_CONDA_PKG_DIRS}
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen headers not found. Please install Eigen3 or set EIGEN3_INCLUDE_DIR.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
endif()

add_library(dvs_core STATIC
    src/dvs_base.cpp
    src/camera.cpp
    src/scene_data.cpp
    src/trajectory.cpp
)
target_include_directories(dvs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(dvs_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
target_link_libraries(dvs_core
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)

add_library(dvs_cuda STATIC
    src/cuda/raycast.cu
)
target_include_directories(dvs_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(dvs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(dvs_cuda
    PUBLIC
        CUDA::cudart
)
target_compile_options(dvs_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

add_executable(dvs_simulator
    apps/simulator_main.cpp
)
target_link_libraries(dvs_simulator
    PRIVATE
        dvs_core
        dvs_cuda
)

install(TARGETS dvs_simulator RUNTIME DESTINATION bin)
