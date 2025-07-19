cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /wd4244")
    message(STATUS "Globally disabling warning C4244 to override LibTorch's /WX flag.")
endif()

if(MSVC)
    target_compile_options(xTorch PRIVATE /w34244)
    target_compile_options(xTorch PRIVATE /WX-)

    message(STATUS "Demoting warning C4244 to level 3 to prevent build failure with /WX.")
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "The configurations to build" FORCE)

# Define the project. The included scripts will configure it.
project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# --- Delegate all build logic to platform-specific files ---
if(WIN32)
    # On Windows, execute the windows.cmake script
    include(cmake/windows.cmake)
elseif(UNIX AND NOT APPLE)
    # On Linux, execute the linux.cmake script
    include(cmake/linux.cmake)
else()
    message(FATAL_ERROR "This project is only configured for Windows and Linux.")
endif()