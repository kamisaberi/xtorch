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
find_package(xTorch REQUIRED)

# Use the correct variable 'xTorch_FOUND' (lowercase 'x')
if(xTorch_FOUND)
    # The xTorch_DIR variable is automatically set by find_package.
    # It points to the directory containing xTorchConfig.cmake,
    # e.g., /usr/local/lib/cmake/xTorch
    message(STATUS "xTorch config directory found at: ${xTorch_DIR}")

    # To find the installation root, we go up a few directories from the config file location.
    # This might require adjustment depending on xTorch's install structure,
    # but /lib/cmake/xTorch is a very common pattern.
    get_filename_component(XTORCH_INSTALL_DIR "${xTorch_DIR}" DIRECTORY) # up to /usr/local/lib/cmake
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local/lib
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local

    message(STATUS "Deduced xTorch installation root: ${XTORCH_INSTALL_DIR}")

    # Now, construct the path to the bundled libtorch from this reliable root path
    set(BUNDLED_LIBTORCH_PATH "${XTORCH_INSTALL_DIR}/include/xtorch/third_party/libtorch")

    # Check if that directory actually exists before adding it
    if(EXISTS "${BUNDLED_LIBTORCH_PATH}")
        message(STATUS "Found bundled libtorch at: ${BUNDLED_LIBTORCH_PATH}")
        list(PREPEND CMAKE_PREFIX_PATH ${BUNDLED_LIBTORCH_PATH})
    else()
        message(FATAL_ERROR "Could not find the bundled libtorch directory at the expected path: ${BUNDLED_LIBTORCH_PATH}")
    endif()
endif()
find_package(Torch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)
#find_package(LibZip REQUIRED)

#find_package(onnxruntime REQUIRED)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)


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
#add_library(my_lib my_lib.cpp)
#
## Make sure your library can find the LibTorch headers.
#target_include_directories(my_lib PUBLIC ${TORCH_INCLUDE_DIRS})

# --- Define the Test Executable ---
# This creates an executable named 'run_tests' from your test file.
add_executable(run_tests
        src/test_activations.cpp
        src/test_dataloader.cpp
        src/test_datasets.cpp
        src/test_dropouts.cpp
        src/test_losses.cpp
        src/test_models.cpp
        src/test_normalizations.cpp
        src/test_optimizers.cpp
        src/test_transforms.cpp
)

# Link the test executable against your library, LibTorch, and Google Test.
# - my_lib: Your own code that you are testing.
# - ${TORCH_LIBRARIES}: The required LibTorch libraries.
# - GTest::gtest_main: The Google Test library that includes a main() function.
target_link_libraries(run_tests
        PRIVATE
        xTorch::xTorch
#        my_lib
        ${TORCH_LIBRARIES}
        GTest::gtest_main
)

# --- Enable Testing with CTest ---
# This command enables the testing capabilities in CMake.
enable_testing()

# Add the 'run_tests' executable as a test case to CTest.
# This allows you to run all your tests by simply calling 'ctest'.
add_test(NAME MyLibTorchTests COMMAND run_tests)