# cmake/windows.cmake
# ===============================================================
#  Build script for WINDOWS using Visual Studio and vcpkg.
#  (Corrected and Cleaned Version)
# ===============================================================

# NOTE: The project() command is already in the main CMakeLists.txt

# --- 1. Dependency Downloading (with VS 2019 / CUDA 11.8 compatible versions) ---
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/third_party")
set(LIBTORCH_DIR "${DEPS_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${DEPS_DIR}/onnxruntime")

# Using LibTorch/ONNX versions compatible with CUDA 11.8
#set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip")
#set(LIBTORCH_SHA256 "031376821817e81e3a1e94871030999581c7908861d49a46323c938888e24c68")
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.7.1%2Bcu118.zip")
set(LIBTORCH_SHA256 "186AA930C1510482153BE939E573937FC6682AD527BA86500F50266C8F418428")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-gpu-1.16.3.zip")
set(ONNXRUNTIME_SHA256 "1c9ac64696144e138379633e7287968b594966141315995f553f495534125c11")

if (NOT EXISTS "${DEPS_DIR}")
    file(MAKE_DIRECTORY "${DEPS_DIR}")
endif ()
if (NOT EXISTS "${LIBTORCH_DIR}")
    message(STATUS "LibTorch not found locally. Downloading and extracting...")
    string(REGEX MATCH "([^/]+)$" LIBTORCH_FILENAME ${LIBTORCH_URL})
    set(LIBTORCH_ARCHIVE "${DEPS_DIR}/${LIBTORCH_FILENAME}")
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ARCHIVE} EXPECTED_HASH SHA256=${LIBTORCH_SHA256} SHOW_PROGRESS)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ARCHIVE} WORKING_DIRECTORY ${DEPS_DIR})
    file(REMOVE ${LIBTORCH_ARCHIVE})
endif ()
if (NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "ONNX Runtime not found locally. Downloading and extracting...")
    string(REGEX MATCH "([^/]+)$" ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL})
    set(ONNXRUNTIME_ARCHIVE "${DEPS_DIR}/${ONNXRUNTIME_FILENAME}")
    file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256} SHOW_PROGRESS)
    set(ONNXRUNTIME_TEMP_DIR "${DEPS_DIR}/onnxruntime_temp")
    file(MAKE_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE} WORKING_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
    file(GLOB EXTRACTED_DIR "${ONNXRUNTIME_TEMP_DIR}/*")
    file(RENAME ${EXTRACTED_DIR} ${ONNXRUNTIME_DIR})
    file(REMOVE_RECURSE ${ONNXRUNTIME_TEMP_DIR})
    file(REMOVE ${ONNXRUNTIME_ARCHIVE})
endif ()

# --- 2. Build Configuration ---
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

# --- 3. Source File Collection ---
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
file(GLOB_RECURSE MATH_FILES CONFIGURE_DEPENDS src/math/*.cpp)

set(LIBRARY_SOURCE_FILES ${ACTIVATION_FILES} ${BASE_FILES} ${DROPOUT_FILES} ${DATA_LOADER_FILES} ${DATA_PARALLEL_FILES} ${DATASET_FILES} ${LOSS_FILES} ${MODEL_FILES} ${NORMALIZATION_FILES} ${OPTIMIZATION_FILES} ${TRAINER_FILES} ${TRANSFORM_FILES} ${UTILITY_FILES} ${MATH_FILES} path.cpp)

# --- 4. Library Target Definition ---
include_directories(SYSTEM ${CMAKE_SOURCE_DIR})
add_library(xTorch SHARED ${LIBRARY_SOURCE_FILES})


set_target_properties(xTorch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --- 5. Dependency Management for WINDOWS ---

# THE CORE FIX: Append to CMAKE_PREFIX_PATH, don't overwrite it.
# This preserves the paths from the vcpkg toolchain file.
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

# Find all dependencies. CMake will now search BOTH vcpkg paths and the LibTorch path.
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(libzip REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
# LibArchive, LibTar, PkgConfig, and Samplerate are not used on Windows.

# Find ONNX Runtime manually
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)

#set(ENABLE_TESTS OFF CACHE BOOL "Disable libsndfile's internal tests")

# --- Build third-party libs from source ---
#add_subdirectory(third_party/sndfile)
#set_target_properties(sndfile PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(imgui third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp third_party/imgui/imgui_tables.cpp third_party/imgui/imgui_widgets.cpp third_party/imgui/imgui_demo.cpp)
target_include_directories(imgui PUBLIC third_party/imgui)

add_library(imgui_backend_glfw_gl3 third_party/imgui/backends/imgui_impl_glfw.cpp third_party/imgui/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_backend_glfw_gl3 PUBLIC third_party/imgui third_party/imgui/backends)
target_link_libraries(imgui_backend_glfw_gl3 PUBLIC imgui glfw OpenGL::GL)

add_library(implot third_party/implot/implot.cpp third_party/implot/implot_items.cpp)
target_include_directories(implot PUBLIC third_party/implot third_party/imgui)
target_link_libraries(implot PUBLIC imgui)

# --- 6. Configure Include Directories and Link Libraries ---
target_include_directories(xTorch PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(xTorch PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CURL_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${LibLZMA_INCLUDE_DIRS}
#        ${SndFile_INCLUDE_DIRS}
        ${LIBZIP_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(xTorch
        PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${LIBZIP_LIBRARIES}
        ${ONNXRUNTIME_LIBRARY}
        PRIVATE
        ${CURL_LIBRARIES}
        ${ZLIB_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
#        ${SndFile_LIBRARIES}
#        sndfile
        imgui_backend_glfw_gl3
        implot
)

# --- 7. Installation Configuration ---
include(GNUInstallDirs)
install(TARGETS xTorch EXPORT xTorchTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)
install(FILES xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY python/xtorch_py/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER}/py_modules)
install(FILES python/requirements.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER})
install(EXPORT xTorchTargets FILE xTorchTargets.cmake NAMESPACE xTorch:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xTorchConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)
if (NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()