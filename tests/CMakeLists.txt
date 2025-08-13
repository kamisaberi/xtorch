# Set the minimum required version of CMake.
cmake_minimum_required(VERSION 3.14)

# Set the project name.
project(MyLibTorchProject CXX)

# Set the C++ standard to C++17. This is a good default for modern C++.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find LibTorch ---
# For this to work, you must specify the path to your LibTorch installation
# when running cmake, e.g.,
# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)

find_package(Torch REQUIRED)
#find_package(xTorch REQUIRED)

# [IMPORTANT] LibTorch ABI Compatibility
# The pre-built LibTorch binaries can have ABI compatibility issues with other libraries.
# If you get linker errors related to "std::__cxx11::string" or similar undefined symbols,
# it often means LibTorch was compiled with a different ABI.
# Uncommenting the following line forces the use of the older ABI and can resolve these issues.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# --- Fetch and Configure Google Test ---
# This downloads Google Test from its official GitHub repository and makes it
# available to our project.
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        # It's good practice to pin to a specific version for reproducibility.
        GIT_TAG        release-1.12.1
)

# Set this property to prevent GTest from being built and installed globally.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

## --- Define Your Application Library ---
## This creates a library from your source code in the 'src' directory.
add_library(my_lib my_lib.cpp)
#
## Make sure your library can find the LibTorch headers.
target_include_directories(my_lib PUBLIC ${TORCH_INCLUDE_DIRS})

# --- Define the Test Executable ---
# This creates an executable named 'run_tests' from your test file.
add_executable(run_tests test_linalg.cpp)

# Link the test executable against your library, LibTorch, and Google Test.
# - my_lib: Your own code that you are testing.
# - ${TORCH_LIBRARIES}: The required LibTorch libraries.
# - GTest::gtest_main: The Google Test library that includes a main() function.
target_link_libraries(run_tests
        PRIVATE
        my_lib
        ${TORCH_LIBRARIES}
        GTest::gtest_main
)

# --- Enable Testing with CTest ---
# This command enables the testing capabilities in CMake.
enable_testing()

# Add the 'run_tests' executable as a test case to CTest.
# This allows you to run all your tests by simply calling 'ctest'.
add_test(NAME MyLibTorchTests COMMAND run_tests)