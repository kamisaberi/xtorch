#!/bin/bash

###############################################################################
# Author: Kamran Saberifard
# Email: kamisaberi@gmail.com
# GitHub: https://github.com/kamisaberi
#
# System Prerequisite Installer
# Description:
#   Installs required system-level development libraries for the xTorch project.
#   This script prepares the system. The C++ dependencies (LibTorch, ONNX Runtime)
#   are handled automatically by CMake during the build process.
###############################################################################

set -e # Exit immediately if a command fails

# --- System Detection (Your original code, it's great!) ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    SYSTEM="windows"
elif grep -q "ID=ubuntu" /etc/os-release 2>/dev/null; then
    SYSTEM="ubuntu"
elif grep -q "ID=arch" /etc/os-release 2>/dev/null; then
    SYSTEM="arch"
# ... (rest of your OS detection logic is fine)
else
    # A more robust fallback
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        SYSTEM=$ID
    else
        echo "Unsupported system"
        exit 1
    fi
fi
echo "Detected system: $SYSTEM"


# --- Package Installation Function (Your original code, slightly tweaked for clarity) ---
check_and_install() {
    local pkg_name=$1
    local human_name=$2
    case "$SYSTEM" in
        ubuntu|mint|debian)
            if ! dpkg -s "$pkg_name" >/dev/null 2>&1; then
                echo "Installing $human_name ($pkg_name)..."
                sudo apt-get install -y "$pkg_name"
            else
                echo "$human_name ($pkg_name) is already installed."
            fi
            ;;
        arch|manjaro)
            if ! pacman -Q "$pkg_name" >/dev/null 2>&1; then
                echo "Installing $human_name ($pkg_name)..."
                sudo pacman -S --noconfirm "$pkg_name"
            else
                echo "$human_name ($pkg_name) is already installed."
            fi
            ;;
        # ... Add other OS cases here if needed (Fedora, SUSE, etc.)
        macos)
            if ! brew list "$pkg_name" >/dev/null 2>&1; then
                echo "Installing $human_name ($pkg_name)..."
                brew install "$pkg_name"
            else
                echo "$human_name ($pkg_name) is already installed."
            fi
            ;;
        windows)
            echo "Windows: Please use Chocolatey or vcpkg to install '$human_name' ($pkg_name) manually."
            # Automated Windows installs are complex, manual instructions are safer here.
            ;;
    esac
}

# --- Package Mapping Table ---
# This maps a human-readable name to its package name on different systems.
declare -A packages
packages["OpenCV"]="opencv:opencv:opencv-devel:opencv-devel:opencv:opencv"
packages["HDF5"]="libhdf5-dev:hdf5:hdf5-devel:hdf5-devel:hdf5:hdf5"
packages["cURL"]="libcurl4-openssl-dev:curl:libcurl-devel:libcurl-devel:curl:curl"
packages["ZLib"]="zlib1g-dev:zlib:zlib-devel:zlib-devel:zlib:zlib"
packages["OpenSSL"]="libssl-dev:openssl:openssl-devel:libopenssl-devel:openssl:openssl"
packages["LibLZMA"]="liblzma-dev:xz:xz-devel:xz-devel:xz:xz-utils"
packages["LibArchive"]="libarchive-dev:libarchive:libarchive-devel:libarchive-devel:libarchive:libarchive"
packages["LibTar"]="libtar-dev:libtar:libtar-devel:libtar-devel:libtar:libtar"
packages["LibZip"]="libzip-dev:libzip:libzip-devel:libzip-devel:libzip:libzip"
packages["GLFW3"]="libglfw3-dev:glfw-x11:glfw-devel:libglfw3-devel:glfw:glfw"
packages["LibEigen3"]="libeigen3-dev:libeigen3-dev:libeigen3-dev:libeigen3-dev:libeigen3-dev"


# --- Pre-Installation Checks (Your logic is good) ---
case "$SYSTEM" in
    ubuntu|mint|debian) sudo apt-get update ;;
    arch|manjaro) sudo pacman -Syu --noconfirm ;;
    macos)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew update
        ;;
esac

# --- Main Installation Loop ---
# This now only installs system packages.
echo "Installing required system packages..."
for name in "${!packages[@]}"; do
    IFS=':' read -r ubuntu_pkg arch_pkg fedora_pkg suse_pkg macos_pkg windows_pkg <<< "${packages[$name]}"
    case "$SYSTEM" in
        ubuntu|mint|debian) check_and_install "$ubuntu_pkg" "$name" ;;
        arch|manjaro)       check_and_install "$arch_pkg" "$name" ;;
        macos)              check_and_install "$macos_pkg" "$name" ;;
        # ... Add other OS cases here ...
        *) echo "Skipping '$name' installation for unsupported system '$SYSTEM'." ;;
    esac
done

# --- REMOVED ---
# The entire "LibTorch Installation and CMakeLists.txt Update" section has been removed.
# This logic is fragile and belongs inside CMake itself.

# --- FINAL INSTRUCTIONS ---
echo ""
echo "========================================================================"
echo "System prerequisites have been installed."
echo "The project is now ready to be built."
echo ""
echo "Next steps:"
echo "  1. mkdir build"
echo "  2. cd build"
echo "  3. cmake .."
echo "     (This will automatically download LibTorch and ONNX Runtime)"
echo "  4. make -j\$(nproc)"
echo "========================================================================"