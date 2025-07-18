# cmake/windows.cmake
# ===============================================================
#  Windows-Specific Build Configuration for xTorch
# ===============================================================
message(STATUS "Loading Windows-specific configuration...")

# --- 1. Set vcpkg Toolchain ---
# For this to work without passing -D on the command line, vcpkg must be
# at this path. For more flexibility, passing it via -D is recommended.
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "Vcpkg toolchain file")
message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")


# --- 2. Define Dependency URLs and Download for Windows ---
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.1%2Bcu128.zip")
set(LIBTORCH_SHA256 "B1D8287A7414073E9C6F58327EE7214E3BB96A214A128F80F0FD6EAC81AAFEB4")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip")
set(ONNXRUNTIME_SHA256 "5B5241716B2628C1AB5E79EE620BE767531021149EE68F30FC46C16263FB94DD")
# Trigger the download using the common logic
include(dependencies.cmake)


# --- 3. Find All Windows Dependencies (via vcpkg) ---
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
# LibArchive is NOT found on Windows, as requested.
find_package(libzip REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
# libtar and libsamplerate are not typically used on Windows; vcpkg would be the source if needed.


# --- 4. Link Libraries to xTorch Target ---
message(STATUS "Linking Windows libraries...")
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
        # No LibArchive here
        sndfile # Built from source in targets.cmake
        imgui_backend_glfw_gl3
        implot
)