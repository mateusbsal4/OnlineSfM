cmake_minimum_required(VERSION 2.8)
project(first_version)

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(xtensor REQUIRED)
find_package(xtensor)
find_package(xtensor-blas)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${xtensor_INCLUDE_DIRS})


add_executable(first_version pipeline.cpp )

target_link_libraries(first_version ${OpenCV_LIBS} )
target_link_libraries(first_version Eigen3::Eigen)
target_link_libraries(first_version PUBLIC xtensor)
add_definitions(-DHAVE_CBLAS=1)

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

target_link_libraries(first_version ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas")

target_compile_options(first_version PRIVATE -march=native)


