# 设置可执行文件输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tools)

# for albedo map
find_package(GDAL REQUIRED)
add_executable(albedo_map ${CMAKE_CURRENT_SOURCE_DIR}/main_albedo_map.cu)
target_include_directories(albedo_map
    PRIVATE
        ${GDAL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(albedo_map
    PRIVATE
        ${GDAL_LIBRARIES}
        dvs_core
        dvs_cuda
)
