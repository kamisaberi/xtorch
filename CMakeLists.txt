cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

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