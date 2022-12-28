cmake_minimum_required(VERSION 2.8)
project(first_version)

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(xtensor REQUIRED)  
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${xtensor_INCLUDE_DIRS})


add_executable(first_version pipeline.cpp )

target_link_libraries(first_version ${OpenCV_LIBS} )
target_link_libraries(first_version Eigen3::Eigen)
target_link_libraries(first_version PUBLIC xtensor)

target_compile_options(first_version PRIVATE -march=native)


