cmake_minimum_required(VERSION 3.5.0)
project(test_ceres VERSION 0.1.0 LANGUAGES C CXX)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(test_ceres main.cpp)

target_link_libraries(test_ceres ${CERES_LIBRARIES} ${OpenCV_LIBS} gflags glog)