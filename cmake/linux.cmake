# cmake/linux.cmake
# ===============================================================
#  Linux-Specific Build Configuration for xTorch
# ===============================================================
message(STATUS "Loading Linux-specific configuration...")

# --- 1. Set Linux-Specific Definitions and Flags ---
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(xTorch PROPERTIES LINK_FLAGS "-Wl,--no-undefined")


# --- 2. Define Dependency URLs and Download for Linux ---
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip")
set(LIBTORCH_SHA256 "ae513b437ae99150744ef1d06b02a4ecbbb9275c9ffe540c88909623e3293041")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz")
set(ONNXRUNTIME_SHA256 "2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55")
# Trigger the download using the common logic
include(dependencies.cmake)


# --- 3. Find All Linux Dependencies ---
# On Linux, find_package searches system paths and uses pkg-config.
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED) # Found ONLY on Linux
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_library(LIBTAR_LIBRARY tar REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)
pkg_check_modules(LIBZIP REQUIRED libzip)


# --- 4. Workaround for GCC compiler bug with ImGui ---
if(CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(imgui PRIVATE $<$<CONFIG:Release>:-O0>)
    message(STATUS "Adding -O0 for ImGui in Release builds to work around compiler bug.")
endif()


# --- 5. Link Libraries to xTorch Target ---
message(STATUS "Linking Linux libraries...")
target_link_libraries(xTorch
        PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${LIBZIP_LIBRARIES}
        ${ONNXRUNTIME_LIBRARY}

        PRIVATE
        ${CURL_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES} # Linked ONLY on Linux
        ${SAMPLERATE_LIBRARIES}
        sndfile # Built from source in targets.cmake
        imgui_backend_glfw_gl3
        implot
        pthread
        dl
)

# --- 6. Add Linux-specific post-install script ---
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