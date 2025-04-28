#!/bin/bash

###############################################################################
# Author: Kamran Saberifard
# Email: kamisaberi@gmail.com
# GitHub: https://github.com/kamisaberi
#
# System Dependency Installer Script
# Description:
#   Automatically detects Linux distribution and installs required development
#   libraries for various multimedia and machine learning dependencies.
# Supported Distributions:
#   - Ubuntu, Linux Mint (Debian-based)
#   - Arch Linux, Manjaro (Arch-based)
###############################################################################

# -----------------------------------------------------------------------------
# Distribution Detection
# -----------------------------------------------------------------------------
# Detect Linux distribution by parsing /etc/os-release
# Sets DISTRO variable to: ubuntu, mint, arch, or manjaro
# Exits with error if unsupported distribution is detected
# -----------------------------------------------------------------------------
if grep -q "ID=ubuntu" /etc/os-release; then
    DISTRO="ubuntu"
elif grep -q "ID=arch" /etc/os-release; then
    DISTRO="arch"
elif grep -q "ID=manjaro" /etc/os-release; then
    DISTRO="manjaro"
elif grep -q "ID=linuxmint" /etc/os-release; then
    DISTRO="mint"
else
    echo "Unsupported distribution"
    exit 1
fi

echo "Detected distribution: $DISTRO"

# -----------------------------------------------------------------------------
# Package Installation Function
# -----------------------------------------------------------------------------
# Parameters:
#   $1 - Package name (distribution-specific)
#   $2 - Installation command (for logging purposes)
# Functionality:
#   - Checks if package is already installed
#   - Installs package using appropriate package manager
#   - Supports apt (Debian) and pacman (Arch) package managers
# -----------------------------------------------------------------------------
check_and_install() {
    local package=$1
    local install_cmd=$2

    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "mint" ]; then
        # Debian/Ubuntu package check
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            echo "Package $package not found, installing..."
            sudo apt-get install -y "$package"
        else
            echo "Package $package is already installed"
        fi
    elif [ "$DISTRO" = "arch" ] || [ "$DISTRO" = "manjaro" ]; then
        # Arch Linux package check
        if ! pacman -Q "$package" >/dev/null 2>&1; then
            echo "Package $package not found, installing..."
            sudo pacman -S --noconfirm "$package"
        else
            echo "Package $package is already installed"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Package Mapping Table
# -----------------------------------------------------------------------------
# Maps Ubuntu/Mint package names to their Arch/Manjaro equivalents
# Format: [Ubuntu Package]=Arch Package
# Key Dependencies:
#   - libopencv-dev: OpenCV computer vision library
#   - libhdf5-dev: HDF5 data format support (for OpenCV)
#   - libcurl4-openssl-dev: HTTP client library
#   - zlib1g-dev: Compression library
#   - libarchive-dev: Multi-format archive library
# -----------------------------------------------------------------------------
declare -A packages
packages["libcurl4-openssl-dev"]="curl"
packages["libopencv-dev"]="opencv"
packages["zlib1g-dev"]="zlib"
packages["libssl-dev"]="openssl"
packages["liblzma-dev"]="xz"
packages["libarchive-dev"]="libarchive"
packages["libtar-dev"]="libtar"
packages["libzip-dev"]="libzip"
packages["libsndfile1-dev"]="libsndfile"
packages["libhdf5-dev"]="hdf5"  # Required for OpenCV's HDF5 support

# -----------------------------------------------------------------------------
# Main Installation Loop
# -----------------------------------------------------------------------------
# Iterates through package list and installs distribution-specific packages
# Handles Debian-based and Arch-based distributions differently
# -----------------------------------------------------------------------------
for ubuntu_pkg in "${!packages[@]}"; do
    arch_pkg="${packages[$ubuntu_pkg]}"
    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "mint" ]; then
        check_and_install "$ubuntu_pkg" "apt-get install -y"
    else
        check_and_install "$arch_pkg" "pacman -S --noconfirm"
    fi
done

# -----------------------------------------------------------------------------
# LibTorch Special Handling
# -----------------------------------------------------------------------------
# LibTorch cannot be installed via system package managers
# Provides manual installation instructions
# -----------------------------------------------------------------------------
echo "LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/"