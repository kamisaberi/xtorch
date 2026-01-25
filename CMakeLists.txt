cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
set(CMAKE_C_COMPILER "gcc-12")
set(CMAKE_CXX_COMPILER "g++-12")
set(CMAKE_CUDA_HOST_COMPILER "g++-12")
set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# ===================================================================
# == NEW STRATEGY: Use CUDNN_ROOT and a direct find_package call ==
# This tells CMake's own FindCUDNN module where the root of the
# cuDNN installation is. We do this BEFORE calling find_package(Torch).
# ===================================================================
#set(CUDNN_ROOT "/usr/local/cuda-12.8")
#
## Now we run CMake's own finder first. If this fails, the problem
## is with the system, not with LibTorch. It gives a better error.
#find_package(CUDNN REQUIRED)
## ===================================================================
#
#set(CUDNN_LIBRARY /usr/local/cuda-12.8/lib64/libcudnn.so)
#set(CUDNN_INCLUDE_DIR /usr/local/cuda-12.8/include)


message(STATUS "Manually setting cuDNN paths to /usr/local/cuda-12.8...")
set(CUDNN_INCLUDE_DIR "/usr/local/cuda-12.8/include")
set(CUDNN_LIBRARY "/usr/local/cuda-12.8/lib64/libcudnn.so")
# For good measure, let's also set the Caffe2-specific variable
set(Caffe2_CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR})
set(Caffe2_CUDNN_LIBRARIES ${CUDNN_LIBRARY})


set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "The configurations to build" FORCE)

# Check CPU architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    message(STATUS "Detected Intel-based CPU (Architecture: ${CMAKE_SYSTEM_PROCESSOR})")
    include(cmake/amd.cmake)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    message(STATUS "Detected ARM-based CPU (Architecture: ${CMAKE_SYSTEM_PROCESSOR})")
    include(cmake/arm.cmake)
else()
    message(WARNING "Unknown CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()