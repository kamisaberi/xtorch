cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(xTorch VERSION 0.2.0 LANGUAGES CXX)
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