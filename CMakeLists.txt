cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
#cmake_policy(SET CMP0002 NEW)

#project(TorchExtension LANGUAGES CXX)

#TODO ----vvvv---- TorchExtension MODULE DEFINITION ----vvvv----

project(TorchExtension VERSION 0.1.0 LANGUAGES CXX)

#add_executable( ${CMAKE_PROJECT_NAME} main.cpp src/datasets/cifar100.cpp src/datasets/cifar10.cpp src/datasets/fashion-mnist.cpp)
add_library(${CMAKE_PROJECT_NAME} SHARED
        src/datasets/cifar100.cpp
        src/datasets/cifar10.cpp
        src/datasets/fashion-mnist.cpp
        src/utils/archiver.cpp
        src/utils/downloader.cpp
)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)



if (NOT LIBTAR_LIBRARY OR NOT LIBTAR_INCLUDE_DIR)
    message(FATAL_ERROR "libtar not found. Please install libtar (e.g., sudo apt-get install libtar-dev).")
endif ()

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)


find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
)


#TODO ----vvvv---- EXPORT MODULE  ----vvvv----

set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

#set(SOURCE_FILES src/library.cpp)

#add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1)

install(TARGETS ${PROJECT_NAME} EXPORT TorchExtensionConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT TorchExtensionConfig DESTINATION share/TorchExtension/cmake)

export(TARGETS ${PROJECT_NAME} FILE TorchExtensionConfig.cmake)



#message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
#message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
#message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")


#TODO ----vvvv---- TEST PROJECT ----vvvv----

add_subdirectory(examples/test)

#cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
#project(Test)
#list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
#find_package(Torch REQUIRED)
#find_package(TorchExtension REQUIRED)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#add_executable(${CMAKE_PROJECT_NAME} main.cpp)
#target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
#set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

