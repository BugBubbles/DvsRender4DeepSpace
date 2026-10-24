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
option(ENABLE_PHYSIC_EFFECTS "Enable electron shot noise, dark current, mismatch, reset and leakage" ON)
if (ENABLE_PHYSIC_EFFECTS)
    add_compile_definitions(ENABLE_PHYSIC_EFFECTS)
else()
    message(WARNING "Physical effects are disabled.")
endif()

option(ENABLE_AER_DELAY "Enable AER delay for simulating the transmission delay of events" ON)
if (ENABLE_AER_DELAY)
    add_compile_definitions(ENABLE_AER_DELAY)
else()
    message(WARNING "Generic readout queue is disabled.")
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

option(BUILD_SENSOR_TESTS "Build physical sensor regression tests" OFF)
if(BUILD_SENSOR_TESTS)
    enable_testing()
    add_executable(test_sensor tests/test_sensor.cu)
    target_link_libraries(test_sensor PRIVATE dvs_core dvs_cuda CUDA::cudart)
    add_test(NAME physical_sensor COMMAND test_sensor)
endif()
if(BUILD_SENSOR_TESTS)
    add_executable(test_writer tests/test_writer.cpp)
    target_include_directories(test_writer PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(test_writer PRIVATE dvs_core CUDA::cudart ${HDF5_LIBRARIES})
    add_test(NAME event_writer COMMAND test_writer)
    add_executable(test_frames tests/test_frames.cpp)
    target_include_directories(test_frames PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(test_frames PRIVATE dvs_core CUDA::cudart ${HDF5_LIBRARIES})
    target_compile_definitions(test_frames PRIVATE DVS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME frame_writer COMMAND test_frames)
    add_executable(test_dataset_budget tests/test_dataset_budget.cpp)
    target_include_directories(test_dataset_budget PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(test_dataset_budget PRIVATE dvs_core CUDA::cudart ${HDF5_LIBRARIES})
    add_test(NAME dataset_budget COMMAND test_dataset_budget)
    add_executable(test_repeated_trajectory tests/test_repeated_trajectory.cpp)
    target_link_libraries(test_repeated_trajectory PRIVATE dvs_core CUDA::cudart)
    target_compile_definitions(test_repeated_trajectory PRIVATE DVS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME repeated_trajectory COMMAND test_repeated_trajectory)
endif()

if(BUILD_SENSOR_TESTS)
    add_executable(test_render tests/test_render.cu)
    target_link_libraries(test_render PRIVATE dvs_core dvs_cuda CUDA::cudart)
    add_test(NAME render_davis346 COMMAND test_render ${CMAKE_CURRENT_SOURCE_DIR}/config/davis346.yaml ${CMAKE_CURRENT_BINARY_DIR}/validation/davis346)
    add_test(NAME render_prophesee COMMAND test_render ${CMAKE_CURRENT_SOURCE_DIR}/config/prophesee.yaml ${CMAKE_CURRENT_BINARY_DIR}/validation/prophesee)
    target_compile_definitions(test_sensor PRIVATE DVS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(BUILD_SENSOR_TESTS)
    add_executable(test_raster tests/test_raster.cu)
    target_link_libraries(test_raster PRIVATE dvs_core dvs_cuda CUDA::cudart)
    add_test(NAME raster_scene_inputs COMMAND test_raster ${CMAKE_CURRENT_BINARY_DIR}/validation/raster)
endif()
