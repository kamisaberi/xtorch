
# xTorch Installation Guide

## Supported Operating Systems
xTorch supports the following Linux distributions:
- **Ubuntu**
- **Linux Mint**
- **Manjaro (Arch-based)**

## Step 1: Install Required Packages

### On Ubuntu / Linux Mint
```bash
sudo apt-get update
sudo apt-get install -y libcurl4-openssl-dev libopencv-dev zlib1g-dev libssl-dev \
    liblzma-dev libarchive-dev libtar-dev libzip-dev libsndfile1-dev \
    build-essential cmake git
```

### On Manjaro / Arch
```bash
sudo pacman -Syu --needed curl opencv zlib openssl xz libarchive libtar libzip libsndfile base-devel cmake git
```

---

## Step 2: Download and Install LibTorch (PyTorch C++)

1. Go to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Choose:
    - Stable
    - Linux
    - Language: **C++/Java**
    - Compute Platform: CPU or CUDA
3. Download and extract the `libtorch` archive:
```bash
unzip libtorch-cxx11-abi-shared-with-deps-*.zip -d ~/libtorch
```

### Optional: Set Environment Variables
```bash
export CMAKE_PREFIX_PATH=~/libtorch/libtorch
export LD_LIBRARY_PATH=~/libtorch/libtorch/lib:$LD_LIBRARY_PATH
```

---

## Step 3: Build xTorch

### Clone the repo
```bash
git clone <your-xTorch-repo-url>
cd xtorch
```

### Create a build directory and configure
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch/libtorch -DCMAKE_BUILD_TYPE=Release ..
```

### Compile
```bash
make -j$(nproc)
```

---

## Step 4: Install xTorch

```bash
sudo make install
sudo ldconfig
```

---

## Step 5: Use xTorch in Your Project

### With CMake
```cmake
find_package(xTorch REQUIRED)
target_link_libraries(MyApp PRIVATE xTorch::xTorch)
```

### Manually (if no package config)
```cmake
target_include_directories(MyApp PRIVATE /usr/local/include/xtorch)
target_link_libraries(MyApp PRIVATE /usr/local/lib/libxTorch.so)
```

---

## Notes

- Make sure all dependencies are installed.
- Recompile xTorch if you upgrade major dependencies.
- Use the correct ABI version of LibTorch for your system (C++11 recommended).

Enjoy building deep learning apps in C++ with **xTorch**!
