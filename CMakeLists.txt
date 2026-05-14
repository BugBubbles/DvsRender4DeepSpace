cmake_minimum_required(VERSION 3.22)
project(dvs_simulator_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH "/usr/local;/usr;${CMAKE_PREFIX_PATH}")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()
file(GLOB BVH_SOURCES "src/bvh/*.cpp")
file(GLOB BASIC_SOURCES "src/*.cpp")
add_library(dvs_core STATIC
    ${BASIC_SOURCES}
    ${BVH_SOURCES}
)
target_include_directories(dvs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(dvs_core
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)

file (GLOB_RECURSE CUDA_SOURCES "src/cuda/*.cu")
add_library(dvs_cuda STATIC
    ${CUDA_SOURCES}
)
target_include_directories(dvs_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dvs_cuda
    PUBLIC
        CUDA::cudart
        Eigen3::Eigen
)

add_executable(dvs_simulator
    apps/simulator_main.cpp
)
target_link_libraries(dvs_simulator
    PRIVATE
        dvs_core
        dvs_cuda
)