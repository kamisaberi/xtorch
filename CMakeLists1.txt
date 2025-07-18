# ===============================================================
# xTorch | Top-Level CMake Configuration
#
# This file defines the project and delegates all build logic to
# platform-specific and feature-specific scripts.
# ===============================================================

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Define the project. This must be the first command.
project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# Set the C++ standard for the entire project.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Include all build logic from the 'cmake' directory ---

# 1. Define all library targets (xTorch, ImGui, etc.).
#    This is done first so platform scripts can modify them.
include(cmake/targets.cmake)

# 2. Configure the build environment for the specific OS.
#    This script finds and links all necessary dependencies.
if(WIN32)
    include(cmake/windows.cmake)
elseif(UNIX AND NOT APPLE)
    include(cmake/linux.cmake)
else()
    message(FATAL_ERROR "Unsupported Operating System: ${CMAKE_SYSTEM_NAME}")
endif()

# 3. Add common installation rules. Platform-specific install
#    rules are handled in their respective files.
include(cmake/installation.cmake)