#!/bin/bash

# Detect distribution
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

# Function to check and install package
check_and_install() {
    local package=$1
    local install_cmd=$2

    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "mint" ]; then
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            echo "Package $package not found, installing..."
            sudo apt-get install -y "$package"
        else
            echo "Package $package is already installed"
        fi
    elif [ "$DISTRO" = "arch" ] || [ "$DISTRO" = "manjaro" ]; then
        if ! pacman -Q "$package" >/dev/null 2>&1; then
            echo "Package $package not found, installing..."
            sudo pacman -S --noconfirm "$package"
        else
            echo "Package $package is already installed"
        fi
    fi
}

# List of packages (Ubuntu/Mint package, Arch/Manjaro package)
declare -A packages
packages["libcurl4-openssl-dev"]="curl"
packages["libopencv-dev"]="opencv"
packages["zlib1g-dev"]="zlib"
packages["libssl-dev"]="openssl"
packages["liblzma-dev"]="xz"
packages["libarchive-dev"]="libarchive"
packages["libtar-dev"]="libtar"
packages["libzip-dev"]="libzip"

# Check and install each package
for ubuntu_pkg in "${!packages[@]}"; do
    arch_pkg="${packages[$ubuntu_pkg]}"
    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "mint" ]; then
        check_and_install "$ubuntu_pkg" "apt-get install -y"
    else
        check_and_install "$arch_pkg" "pacman -S --noconfirm"
    fi
done

# Handle LibTorch separately (not in package managers)
echo "LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/"