cmake_minimum_required(VERSION 2.8)
project(OnlineSfM)

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(xtensor REQUIRED)  
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${xtensor_INCLUDE_DIRS})


add_executable(OnlineSfM pipeline.cpp )

target_link_libraries(OnlineSfM ${OpenCV_LIBS} )
target_link_libraries(OnlineSfM Eigen3::Eigen)
target_link_libraries(OnlineSfM PUBLIC xtensor)

target_compile_options(OnlineSfM PRIVATE -march=native)


