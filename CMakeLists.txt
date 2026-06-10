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
find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()
# 是否启用暗电流和不应期噪声模型
option(ENABLE_PHYSIC_EFFECTS "Enable dark current, noisy threshold and refractory noise model" ON)
if (ENABLE_PHYSIC_EFFECTS)
    add_compile_definitions(ENABLE_PHYSIC_EFFECTS)
else()
    message(WARNING "Physical effects are disabled.")
endif()

option(ENABLE_HIGH_PASS_FILTER "Enable high-pass filter for simulating the response of photoreceptors" ON)
if (ENABLE_PHYSIC_EFFECTS AND ENABLE_HIGH_PASS_FILTER)
    add_compile_definitions(ENABLE_HIGH_PASS_FILTER)
else()
    message(WARNING "High-pass filter is disabled. To use high-pass filter characteristic, ENABLE_PHYSIC_EFFECTS option is needed first.")
endif()

option(ENABLE_AER_DELAY "Enable AER delay for simulating the transmission delay of events" ON)
if (ENABLE_PHYSIC_EFFECTS AND ENABLE_AER_DELAY AND ENABLE_HIGH_PASS_FILTER)
    add_compile_definitions(ENABLE_AER_DELAY)
else()
    message(WARNING "AER delay is disabled. To use AER delay characteristic, both ENABLE_PHYSIC_EFFECTS and ENABLE_HIGH_PASS_FILTER options are needed first.")
endif()

option(BUILD_TOOLS "Build tools for visualization and debugging" OFF)

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
        ${GDAL_INCLUDE_DIRS}
)

target_link_libraries(dvs_core
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${GDAL_LIBRARIES}
)

file (GLOB_RECURSE CUDA_SOURCES "src/*.cu")
add_library(dvs_cuda STATIC
    ${CUDA_SOURCES}
)
target_include_directories(dvs_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}/../targets/x86_64-linux/include
        ${GDAL_INCLUDE_DIRS}
)

target_link_libraries(dvs_cuda
    PUBLIC
        CUDA::cudart
        Eigen3::Eigen
        ${GDAL_LIBRARIES}
)

add_subdirectory(apps)
if (BUILD_TOOLS)
    add_subdirectory(tools/cpp_tools)
endif()
