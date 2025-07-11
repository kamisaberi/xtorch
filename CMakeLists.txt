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

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)



# --- 1. Define Dependency Configuration ---
# All URLs, hashes, and paths are defined here for easy updates.
# NOTE: You MUST update these URLs and hashes for your specific OS and library versions!
# Find URLs from: https://pytorch.org/get-started/locally/ and https://github.com/microsoft/onnxruntime/releases

set(DEPS_DIR "${CMAKE_SOURCE_DIR}/third_party")
set(LIBTORCH_DIR "${DEPS_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${DEPS_DIR}/onnxruntime")

# Example for Linux CPU
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip")
set(LIBTORCH_SHA256 "ae513b437ae99150744ef1d06b02a4ecbbb9275c9ffe540c88909623e3293041")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz")
set(ONNXRUNTIME_SHA256 "2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55")


# Create the third_party directory if it doesn't exist
if(NOT EXISTS "${DEPS_DIR}")
    file(MAKE_DIRECTORY "${DEPS_DIR}")
endif()

# --- 2. Fetch LibTorch if it doesn't exist locally ---
if(NOT EXISTS "${LIBTORCH_DIR}")
    message(STATUS "LibTorch not found locally. Downloading and extracting...")
    set(LIBTORCH_ARCHIVE "${DEPS_DIR}/libtorch.zip")

    # Download the file with a hash check for security
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ARCHIVE}
            EXPECTED_HASH SHA256=${LIBTORCH_SHA256}
            SHOW_PROGRESS)

    # Extract the archive
    message(STATUS "Extracting LibTorch...")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ARCHIVE}
            WORKING_DIRECTORY ${DEPS_DIR}
    )
    # The zip file correctly contains a top-level 'libtorch' folder.

    # Clean up the downloaded archive
    file(REMOVE ${LIBTORCH_ARCHIVE})
    message(STATUS "LibTorch setup complete in ${LIBTORCH_DIR}")
else()
    message(STATUS "Found local LibTorch at ${LIBTORCH_DIR}")
endif()

# --- 3. Fetch ONNX Runtime if it doesn't exist locally ---
if(NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "ONNX Runtime not found locally. Downloading and extracting...")
    # The archive extension can be .zip or .tgz
    string(REGEX MATCH "([^/]+)$" ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL})
    set(ONNXRUNTIME_ARCHIVE "${DEPS_DIR}/${ONNXRUNTIME_FILENAME}")

    file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE}
            EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256}
            SHOW_PROGRESS)

    message(STATUS "Extracting ONNX Runtime...")
    # Extract to a temporary directory first, as the top-level folder name is versioned
    set(ONNXRUNTIME_TEMP_DIR "${DEPS_DIR}/onnxruntime_temp")
    file(MAKE_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE}
            WORKING_DIRECTORY ${ONNXRUNTIME_TEMP_DIR}
    )

    # Find the single created directory inside the temp folder
    file(GLOB EXTRACTED_DIR "${ONNXRUNTIME_TEMP_DIR}/*")
    # Rename it to our desired standard name 'onnxruntime'
    file(RENAME ${EXTRACTED_DIR} ${ONNXRUNTIME_DIR})

    # Clean up the temporary directory and archive
    file(REMOVE_RECURSE ${ONNXRUNTIME_TEMP_DIR})
    file(REMOVE ${ONNXRUNTIME_ARCHIVE})
    message(STATUS "ONNX Runtime setup complete in ${ONNXRUNTIME_DIR}")
else()
    message(STATUS "Found local ONNX Runtime at ${ONNXRUNTIME_DIR}")
endif()


# --- 4. Find and Configure the Libraries (now that they are guaranteed to exist) ---
# This section is identical to the original "local setup" guide.

# Find LibTorch
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
message(STATUS "Configured LibTorch: ${TORCH_VERSION}")

# Find ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_DIR}/lib"
        REQUIRED
)
message(STATUS "Configured ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")
include_directories(${ONNXRUNTIME_INCLUDE_DIR})






# =============================================================================
# Build Configuration
# =============================================================================
# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting default build type to: ${CMAKE_BUILD_TYPE}")
endif ()

# =============================================================================
# Source File Collection
# =============================================================================
# Recursively collect all source files from project directories
# CONFIGURE_DEPENDS enables automatic reconfiguration when files are added/removed


file(GLOB_RECURSE ACTIVATION_FILES CONFIGURE_DEPENDS src/activations/*.cpp)
file(GLOB_RECURSE BASE_FILES CONFIGURE_DEPENDS src/base/*.cpp)
file(GLOB_RECURSE DROPOUT_FILES CONFIGURE_DEPENDS src/dropouts/*.cpp)

file(GLOB_RECURSE DATA_LOADER_FILES CONFIGURE_DEPENDS src/data_loaders/*.cpp)
file(GLOB_RECURSE DATA_PARALLEL_FILES CONFIGURE_DEPENDS src/data_parallels/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)

file(GLOB_RECURSE LOSS_FILES CONFIGURE_DEPENDS src/losses/*.cpp)

file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)

file(GLOB_RECURSE NORMALIZATION_FILES CONFIGURE_DEPENDS src/normalizations/*.cpp)
file(GLOB_RECURSE OPTIMIZATION_FILES CONFIGURE_DEPENDS src/optimizations/*.cpp)

file(GLOB_RECURSE TRAINER_FILES CONFIGURE_DEPENDS src/trainers/*.cpp)
file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)


# Collect all header files from include directory
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS ./include/*.h)






# Combine all source files into single list for library target
set(LIBRARY_SOURCE_FILES
        ${ACTIVATION_FILES}
        ${BASE_FILES}
        ${DROPOUT_FILES}

        ${DATA_LOADER_FILES}
        ${DATA_PARALLEL_FILES}
        ${DATASET_FILES}

        ${LOSS_FILES}

        ${MODEL_FILES}

        ${NORMALIZATION_FILES}
        ${OPTIMIZATION_FILES}

        ${TRAINER_FILES}
        ${TRANSFORM_FILES}
        ${UTILITY_FILES}
        path.cpp
)

# =============================================================================
# Include Directories
# =============================================================================
# Add standard system include paths
message(${CMAKE_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR} /usr/include /usr/local/include)

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

#list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
#if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
#    list(APPEND CMAKE_PREFIX_PATH E:/LIBRARIES/cpp/libtorch)
#elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
#    list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
#elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
#    list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)
#else ()
#    message(WARNING "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
#endif ()

#list(APPEND CMAKE_PREFIX_PATH /proc/cuda/)

# set(SndFile_INCLUDE_DIR "/usr/include")
# set(SndFile_LIBRARY "/usr/lib/x86_64-linux-gnu/libsndfile.so")

#find_package(CUDA REQUIRED)

# Find required dependencies
find_package(Torch REQUIRED)          # PyTorch C++ API
find_package(CURL REQUIRED)           # HTTP client library
find_package(OpenCV REQUIRED)         # Computer vision library
find_package(ZLIB REQUIRED)           # Compression library
find_package(OpenSSL REQUIRED)        # Cryptography and SSL/TLS
find_package(LibLZMA REQUIRED)        # LZMA compression
find_package(LibArchive REQUIRED)     # Multi-format archive library
find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)


# find_package(SndFile REQUIRED)        # Audio file handling


# find_package(onnxruntime REQUIRED)



find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)


find_package(Eigen3 REQUIRED)







## --- 2. Find ONNX Runtime (Local Setup) ---
## Define the path to our local ONNX Runtime include directory
#set(ONNXRUNTIME_INCLUDE_DIR "/home/kami/libs/cpp/onnxruntime-linux-x64-gpu-1.22.0/include")
#
## Find the ONNX Runtime library file.
## The name might be onnxruntime.so (Linux), onnxruntime.dylib (macOS), or onnxruntime.lib (Windows)
#find_library(
#        ONNXRUNTIME_LIBRARY
#        NAMES onnxruntime
#        PATHS "/home/kami/libs/cpp/onnxruntime-linux-x64-gpu-1.22.0/lib"
#        REQUIRED
#)
#message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# Add the ONNX Runtime include directory to the project's include paths

#include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# ====================================================================
#  Fetch and Build libsndfile as part of this project
# ====================================================================

# Include the FetchContent module, which gives us the necessary commands
# include(FetchContent)

# Declare the dependency we want to fetch from its Git repository
# FetchContent_Declare(
#   sndfile                                     # A unique name for this content
#   GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git # The official repo URL
#   GIT_TAG        1.2.2                        # Use a specific, stable release tag for reproducibility!
# )

# This command will download and configure the dependency.
# It makes the targets defined in sndfile's CMakeLists.txt (like 'sndfile')
# available to our project.
# FetchContent_MakeAvailable(sndfile)

# ====================================================================


find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)



add_subdirectory(third_party/sndfile)

# --- ADD THIS LINE ---
# Now, set the property specifically for the 'sndfile' target
set_target_properties(sndfile PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ... the rest of your CMakeLists.txt ...

#add_library(xTorch SHARED ${SOURCES})
#target_link_libraries(xTorch PRIVATE sndfile)




# --- Add ImGui ---
add_library(imgui
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/imgui_demo.cpp
)
target_include_directories(imgui PUBLIC third_party/imgui)

# ImGui Backend for GLFW and OpenGL3
add_library(imgui_backend_glfw_gl3
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_backend_glfw_gl3 PUBLIC third_party/imgui third_party/imgui/backends)

# ======================================================================
# THE FIX IS HERE: We link to the 'glfw' target, not 'glfw3'
target_link_libraries(imgui_backend_glfw_gl3 PUBLIC imgui glfw OpenGL::GL)
# ======================================================================

# --- Add ImPlot ---
add_library(implot
        third_party/implot/implot.cpp
        third_party/implot/implot_items.cpp
)
target_include_directories(implot PUBLIC third_party/implot third_party/imgui)
target_link_libraries(implot PUBLIC imgui)




#find_package(SndFile REQUIRED
#        PATHS /usr/lib/x86_64-linux-gnu/cmake
#        /usr/local/lib/cmake
#        /usr/share/cmake
#        )







# Find libtar (for tar archive handling)
find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)



## Configure include directories for the library
#target_include_directories(xTorch
#        PRIVATE
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#        $<INSTALL_INTERFACE:include>
#        ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}
#        ${ZLIB_INCLUDE_DIRS} ${CURL_INCLUDE_DIR}
#        ${LibArchive_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS})



# ADD THESE TWO NEW BLOCKS INSTEAD
# ==============================================================================

# 1. PUBLIC includes: These are for consumers of your library.
#    This is your library's own public API, located in the 'include' folder.
target_include_directories(xTorch
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 2. PRIVATE includes: These are only needed to build xTorch itself.
#    This prevents local build paths (like for LibTorch) from "leaking"
#    into the public interface, which fixes the error.
target_include_directories(xTorch
        PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR} # This was missing from this section before
        ${CURL_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${LibLZMA_INCLUDE_DIRS}
        ${LibArchive_INCLUDE_DIRS}
        ${SndFile_INCLUDE_DIRS}
        ${LIBTAR_INCLUDE_DIR}
        ${LIBZIP_INCLUDE_DIRS}  # Add this line
)
# ==============================================================================





## Link all required libraries to xTorch
#target_link_libraries(xTorch PRIVATE
#        ${TORCH_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBRARIES}
#        ${LIBTAR_LIBRARY} OpenSSL::SSL OpenSSL::Crypto LibLZMA::LibLZMA
#        ${LibArchive_LIBRARIES} ${SndFile_LIBRARIES}
#        sndfile
#        ${ONNXRUNTIME_LIBRARY}
#        # ${ONNXRUNTIME_LIBRARY}
#        imgui_backend_glfw_gl3 implot
#        PUBLIC
#        ${LIBZIP_LIBRARIES}  # Add this line
#)

target_link_libraries(xTorch
        PUBLIC
        ${TORCH_LIBRARIES}    # Almost certainly public
        ${OpenCV_LIBS}        # Likely public if you use cv::Mat in headers
        ${LIBZIP_LIBRARIES}   # This is the direct cause of your error.
        ${ONNXRUNTIME_LIBRARY}# Likely public if you expose ONNX features

        PRIVATE
        # These are likely implementation details not exposed in your headers.
        ${CURL_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${SndFile_LIBRARIES}
        sndfile
        imgui_backend_glfw_gl3
        implot
        ${EIGEN3_INCLUDE_DIRS}
)


# --- Force Linker to Ignore Errors (THE CORRECT COMMAND) ---
# This sets the general linking flags for the target.
# We use a special syntax to pass the flag directly to the linker.
#set_target_properties(xTorch PROPERTIES
#        LINK_FLAGS "-Wl,--unresolved-symbols=ignore-all"
#)
set_target_properties(xTorch PROPERTIES
        LINK_FLAGS "-Wl,--no-undefined"
)

target_link_options(xTorch PRIVATE "-Wl,--no-undefined")


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
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

)


# Convert project name to lowercase for installation paths
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)


# Install headers and third-party files
install(FILES xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})


install(DIRECTORY python/xtorch_py/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER}/py_modules)
install(FILES python/requirements.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER})


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
if (NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()


# === ADD THIS BLOCK START ===
#
# This block executes the setup.sh script during the 'make install' process.
# It uses install(CODE ...) to run a CMake command at install time.
#
# The execute_process command runs an external script. We pass
# ${CMAKE_INSTALL_PREFIX} as a command-line argument to the script so it
# knows where to create the venv.
#
install(CODE "
    message(STATUS \"Executing post-install script: setup.sh\")
    execute_process(
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup.sh \"${CMAKE_INSTALL_PREFIX}\"
        RESULT_VARIABLE res
    )
    if(NOT res EQUAL 0)
        message(FATAL_ERROR \"Post-install script failed with exit code \${res}\")
    endif()
"
)
# === ADD THIS BLOCK END ===


# =============================================================================
# Subdirectories
# =============================================================================
# Add examples subdirectory (contains demonstration code)
# add_subdirectory(examples)
# Temporary code directory (currently commented out)
#add_subdirectory(temp)
