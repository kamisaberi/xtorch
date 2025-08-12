# Installation Guide for xTorch on Ubuntu

This guide provides instructions for setting up the necessary dependencies, building, and installing the xTorch project on **Ubuntu**.

**Important:** Before you proceed with the build process, you must install the required system-level development libraries and Python environment.

---

## 1. Prerequisites: Install System Dependencies

First, you need to install several libraries and tools that xTorch depends on. Open a terminal and run the following commands to install the required packages using `apt-get`.

### Step 1: Update Package Lists

Ensure your package manager has the latest list of available packages:
```bash
sudo apt-get update
```

### Step 2: Install Core Build Tools

You will need `build-essential` for a C++ compiler (like g++) and other essential tools, and `cmake` to configure and build the project.
```bash
sudo apt-get install -y build-essential cmake
```

### Step 3: Install Python and Project-Specific Libraries

Install Python, the virtual environment module (`venv`), and the development headers for all other required libraries with a single command:
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

## 2. Building the Project

### Step 1: Clone Repository
First clone repository using git command :
```bash
git clone https://github.com/kamisaberi/xtorch
cd xtorch 
```


Once all system dependencies are installed, you can build the xTorch project using CMake. The `CMakeLists.txt` file is configured to automatically download the correct versions of **LibTorch** and **ONNX Runtime**, so you do not need to install them manually.

### Step 1: Create a Build Directory

It is a best practice to create a separate directory for the build files to keep the main project folder clean.

```bash
mkdir build
cd build
```

### Step 2: Configure the Project with CMake

Run `cmake` from inside the `build` directory. This command will find the system libraries you just installed and download the remaining C++ dependencies (LibTorch, ONNX Runtime).

```bash
cmake ..
```

### Step 3: Compile the Project with Make

Use the `make` command to compile the xTorch library. The `-j$(nproc)` flag tells `make` to use all available CPU cores, which significantly speeds up the compilation process.

```bash
make -j$(nproc)
```

After this step, the compiled shared library (`libxTorch.so`) will be located inside your `build` directory.

---

## 3. Installing the Library

After successfully compiling the library, you can install it system-wide. This will copy the library files, headers, and CMake configuration files to standard system directories, making it easy for other projects to find and link against xTorch.

From within the `build` directory, run the following command:

```bash
sudo make install -j$(nproc)
```

This completes the installation process. xTorch is now ready to be used in your C++ applications.