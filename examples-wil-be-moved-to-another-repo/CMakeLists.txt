cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(Examples)

list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(test)

#add_executable(${PROJECT_NAME} main.cpp)
#
#target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
#set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#message(STATUS "CMAKE_PROJECT_NAME:${CMAKE_PROJECT_NAME}")
#message(STATUS "PROJECT_NAME:${PROJECT_NAME}")
