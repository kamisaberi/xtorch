# =============================================================================
# xTorch CMake Build System
# Author: Kamran Saberifard
# Email: kamisaberi@gmail.com
# GitHub: https://github.com/kamisaberi
#
# Project: xTorch Machine Learning Framework
# Version: 0.2.0
# Description: CMake configuration for building xTorch shared library
# =============================================================================

# Minimum required CMake version with FATAL_ERROR if not met
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Project definition with version and language (C++)
project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# =============================================================================
# Build Configuration
# =============================================================================
# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting default build type to: ${CMAKE_BUILD_TYPE}")
endif()

# =============================================================================
# Source File Collection
# =============================================================================
# Recursively collect all source files from project directories
# CONFIGURE_DEPENDS enables automatic reconfiguration when files are added/removed
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS src/models/cnn/*.cpp)
file(GLOB_RECURSE DATA_LOADER_FILES CONFIGURE_DEPENDS src/data_loaders/*.cpp)
file(GLOB_RECURSE DATA_PARALLEL_FILES CONFIGURE_DEPENDS src/data_parallels/*.cpp)
file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)
file(GLOB_RECURSE TRAINER_FILES CONFIGURE_DEPENDS src/trainers/*.cpp)
file(GLOB_RECURSE TEMP_FILES CONFIGURE_DEPENDS src/temp/*.cpp)

# Collect all header files from include directory
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS ./include/*.h)

# Combine all source files into single list for library target
set(LIBRARY_SOURCE_FILES
        ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILES} ${DEFINITION_FILES}
        ${MODEL_FILES} ${MODEL_CNN_FILES} ${DATA_LOADER_FILES}
        ${TRANSFORM_FILES} ${TEMP_FILES} ${HEADER_FILES} ${TRAINER_FILES} ${DATA_PARALLEL_FILES}
)

# =============================================================================
# Include Directories
# =============================================================================
# Add standard system include paths
include_directories(SYSTEM /usr/include /usr/local/include)

# =============================================================================
# Library Target Configuration
# =============================================================================
# Create shared library target with collected source files
add_library(xTorch SHARED ${LIBRARY_SOURCE_FILES})

# Set C++17 standard requirement
set_target_properties(xTorch PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)

# =============================================================================
# Dependency Management
# =============================================================================
# Add custom library paths (LibTorch)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)

# Find required dependencies
find_package(Torch REQUIRED)          # PyTorch C++ API
find_package(CURL REQUIRED)           # HTTP client library
find_package(OpenCV REQUIRED)         # Computer vision library
find_package(ZLIB REQUIRED)           # Compression library
find_package(OpenSSL REQUIRED)        # Cryptography and SSL/TLS
find_package(LibLZMA REQUIRED)        # LZMA compression
find_package(LibArchive REQUIRED)     # Multi-format archive library
find_package(SndFile REQUIRED)        # Audio file handling

# Find libtar (for tar archive handling)
find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)

# Configure include directories for the library
target_include_directories(xTorch
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS} ${CURL_INCLUDE_DIR}
        ${LibArchive_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS})

# Link all required libraries to xTorch
target_link_libraries(xTorch
        ${TORCH_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY} OpenSSL::SSL OpenSSL::Crypto LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES} ${SndFile_LIBRARIES})

# Optional CUDA paths (commented out by default)
list(APPEND DCUDA_TOOLKIT_ROOT_DIR /opt/cuda)
list(APPEND Dnvtx3_dir /opt/cuda/include/nvtx3)

# =============================================================================
# Installation Configuration
# =============================================================================
include(GNUInstallDirs)  # Standard GNU installation directories

# Install library targets
install(TARGETS xTorch EXPORT xTorchTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Convert project name to lowercase for installation paths
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)

# Install headers and third-party files
install(FILES xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})

# CMake package configuration for find_package() support
install(EXPORT xTorchTargets
        FILE xTorchTargets.cmake
        NAMESPACE xTorch::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

# Generate and install version/config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xTorchConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

# =============================================================================
# Uninstall Target
# =============================================================================
if(NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# =============================================================================
# Subdirectories
# =============================================================================
# Add examples subdirectory (contains demonstration code)
add_subdirectory(examples)
# Temporary code directory (currently commented out)
add_subdirectory(temp)