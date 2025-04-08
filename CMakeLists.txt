# -----------------------------------------------------------------------------
# Project Setup for xTorch
# -----------------------------------------------------------------------------
# This CMake configuration file sets up the build environment for the xTorch
# project, a C++ library. It specifies required dependencies, includes, and
# configuration for building the project. It also handles library and
# executable installation.
#
# Version: 0.2.0
# CMake Minimum Version: 3.28
#
# -----------------------------------------------------------------------------
# CMake Configuration
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Define the project name and version
project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------
# Lists all the source files for various components of the project:
#   - Utility files
#   - Media files
#   - Dataset files
#   - Definition files
#   - Model files
#   - Dataloader files
#   - Transform files
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS src/models/cnn/*.cpp)
file(GLOB_RECURSE DATALOADER_FILES CONFIGURE_DEPENDS src/data-loaders/*.tpp)
file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)

# Combine all source files into one variable for easy access
set(LIBRARY_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILIES} ${DEFINITION_FILES} ${MODEL_FILES}  ${DATALOADER_FILES} ${TRANSFORM_FILES})

# -----------------------------------------------------------------------------
# Library Setup
# -----------------------------------------------------------------------------
# Create the shared library from the source files
add_library(${CMAKE_PROJECT_NAME} SHARED ${LIBRARY_SOURCE_FILES})

# -----------------------------------------------------------------------------
# External Dependencies
# -----------------------------------------------------------------------------
# Find required libraries and header files for dependencies such as libtar,
# libtorch, OpenCV, Zlib, OpenSSL, LibLZMA, etc.
find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)

# Ensure that libtar is found, else terminate the build
if (NOT LIBTAR_LIBRARY OR NOT LIBTAR_INCLUDE_DIR)
    message(FATAL_ERROR "libtar not found. Please install libtar (e.g., sudo apt-get install libtar-dev).")
endif ()

# Set C++ standard version for the project
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Add path to libtorch
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)

# Find the necessary libraries and packages for the project
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(SndFile REQUIRED)

# -----------------------------------------------------------------------------
# Include Directories
# -----------------------------------------------------------------------------
# Include directories for required external libraries
include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})

# -----------------------------------------------------------------------------
# Target Include Directories and Link Libraries
# -----------------------------------------------------------------------------
# Specify include directories for the project
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS} ${LIBSNDFILE_INCLUDE_DIRS})

# Link required libraries to the project
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${LIBSNDFILE_LIBRARIES}
)

# -----------------------------------------------------------------------------
# Build Type Configuration
# -----------------------------------------------------------------------------
# Set the default build type to "Release" if none is specified
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# -----------------------------------------------------------------------------
# Install Configuration
# -----------------------------------------------------------------------------
# Include installation directories and configuration
include(GNUInstallDirs)

# Set up include directories for both build and install environments
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Set version and library properties for the target
set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1)

# Install the library and include files
install(TARGETS ${PROJECT_NAME} EXPORT xTorchConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

# Install CMake configuration
install(EXPORT xTorchConfig DESTINATION share/xTorch/cmake)

# Export the target for use in other projects
export(TARGETS ${PROJECT_NAME} FILE xTorchConfig.cmake)

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
# Add subdirectories for example and temporary code
add_subdirectory(examples)
add_subdirectory(temp)
