# Installation

This guide provides a complete walkthrough for installing xTorch and its dependencies on **Ubuntu**.

The process involves three main stages:
1.  **Install System Dependencies**: Set up the required development tools and libraries.
2.  **Build the Project**: Compile the xTorch library from the source code.
3.  **Install the Library**: Make xTorch available system-wide for your C++ projects.

---

## 1. Prerequisites: System Dependencies

Before building xTorch, you must install several essential system-level packages.

### Step 1: Update Package Lists

First, ensure your package manager has the latest list of available software.

```bash
sudo apt-get update
```

### Step 2: Install Core Build Tools

Install `build-essential` (which includes the g++ compiler) and `cmake`.

```bash
sudo apt-get install -y build-essential cmake
```

### Step 3: Install Required Libraries

Install Python and all other libraries required by xTorch with a single command.

```bash
sudo apt-get install -y \
    python3 \
    python3-venv \
    libopencv-dev \
    libhdf5-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    libssl-dev \
    liblzma-dev \
    libarchive-dev \
    libtar-dev \
    libzip-dev \
    libglfw3-dev \
    libeigen3-dev \
    libsamplerate0-dev
```

---

## 2. Clone and Build xTorch

The build process is managed by CMake, which will automatically download and configure **LibTorch** and **ONNX Runtime** for you.

### Step 1: Clone the Repository

Get the latest source code from the official GitHub repository.

```bash
git clone https://github.com/kamisaberi/xtorch
cd xtorch
```

### Step 2: Create a Build Directory

It's standard practice to build the project in a separate directory to keep the source tree clean.

```bash
mkdir build
cd build
```

### Step 3: Configure with CMake

Run `cmake` from the `build` directory. This will check for dependencies and prepare the build files.

```bash
cmake ..
```

!!! note "Custom CUDA Path"
If your NVIDIA CUDA Toolkit is installed in a non-standard location, you'll need to tell CMake where to find the compiler (`nvcc`). First, find the path with `which nvcc`, then run cmake with the following flag:
```bash
cmake -DCMAKE_CUDA_COMPILER='/path/to/your/nvcc' ..
```

### Step 4: Compile the Library

Use `make` to compile the entire project. The `-j$(nproc)` flag uses all available CPU cores to significantly speed up the process.

```bash
make -j$(nproc)
```

Once complete, the compiled shared library (`libxTorch.so`) will be available in the `build` directory.

---

## 3. Install the Library

To make xTorch easily accessible to other C++ projects, install it system-wide. This command copies the header files, the compiled library, and CMake configuration files to standard system locations (like `/usr/local/lib` and `/usr/local/include`).

From within the `build` directory, run:

```bash
sudo make install
```

---

## 4. Verify the Installation

You can verify that everything was built correctly by running the included unit tests and examples.

### Running Unit Tests

1.  Navigate to the `test` directory in the source folder.
    ```bash
    cd ../test
    ```
2.  Create a build directory and configure with CMake. This will download the Google Test framework.
    ```bash
    mkdir build
    cd build
    cmake ..
    ```
3.  Compile and run the tests.
    ```bash
    make
    ./run_test
    ```

### Building and Running Examples

The main repository includes a few key examples to get you started.

1.  Navigate to the main `examples` directory (create it if it doesn't exist or use the one from the repo).
2.  Follow the same build process:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3.  You can now run any of the compiled examples, such as:
    ```bash
    # Run the LeNet MNIST example
    ./classifying_handwritten_digits_with_lenet_on_mnist

    # Run the DCGAN image generation example
    ./generating_images_with_dcgan
    ```

!!! success "Installation Complete!"
xTorch is now successfully installed and ready to be used in your C++ applications. For a comprehensive collection of advanced examples, check out the dedicated [xtorch-examples repository](https://github.com/kamisaberi/xtorch-examples).
