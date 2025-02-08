#############
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TorchExt)
add_executable( ${CMAKE_PROJECT_NAME} main.cpp src/datasets/cifar100.cpp src/datasets/cifar10.cpp)
#add_library(dcgan include/CIFAR100.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)

if(NOT LIBTAR_LIBRARY OR NOT LIBTAR_INCLUDE_DIR)
    message(FATAL_ERROR "libtar not found. Please install libtar (e.g., sudo apt-get install libtar-dev).")
endif()

set_property(TARGET  ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)

find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
#find_package(LIBTAR REQUIRED)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBTAR REQUIRED libtar)
#find_package(Archive REQUIRED)

# Add libtar manually




#find_library(LIBTAR_LIBRARY tar)
#find_path(LIBTAR_INCLUDE_DIR tar.h PATH_SUFFIXES libtar)
## Check if libtar was found
#if(NOT LIBTAR_LIBRARY OR NOT LIBTAR_INCLUDE_DIR)
#    message(FATAL_ERROR "libtar not found. Please install libtar.")
#endif()
#
## Include the directory and link the library
##add_executable(extract_tar extract_tar.cpp)
#target_include_directories(dcgan PRIVATE ${LIBTAR_INCLUDE_DIR})
##target_link_libraries(extract_tar PRIVATE ${LIBTAR_LIBRARY})







include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})

target_include_directories( ${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR})
#target_link_libraries(extract_tar PRIVATE ${LIBTAR_LIBRARY})

#
#link_directories(/usr/lib)

target_link_libraries( ${CMAKE_PROJECT_NAME} ${CURL_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBS} ${ZIP_LIBRARIES}  zip ${ZLIB_LIBRARIES} ${LIBTAR_LIBRARY})
#target_link_libraries(dcgan ${CURL_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBS} ${ZIP_LIBRARIES}  zip ${ZLIB_LIBRARIES} extract_tar ${LIBTAR_LIBRARY})

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
#
#
##################

#cmake_minimum_required(VERSION 3.10)
#project(ZipExtractor)

#set(CMAKE_CXX_STANDARD 11)

# Find ZLIB
#find_package(ZLIB REQUIRED)

# Include directories for libzip
#include_directories(/usr/include)

# Add the executable
#add_executable(dcgan temp/unzip2.cpp)

# Link against libzip and zlib


#vvvvvvvvv CRAAAAAAAAAAAAAAAAAAAAAAAAAP vvvvvvvv

##find_package(Zip REQUIRED)
##find_package(Zip REQUIRED)

#target_link_libraries(dcgan "${TORCH_LIBRARIES}")
#target_link_libraries(dcgan ${OpenCV_LIBS})
#target_link_libraries(dcgan ${ZLIB_LIBS})

#target_link_libraries(dcgan ${ZIP_LIBRARIES})
#target_link_libraries(dcgan PRIVATE zip ${ZLIB_LIBRARIES})
#target_link_libraries(dcgan PRIVATE zip zlib)
#
