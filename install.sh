#!/bin/bash

###############################################################################
# Author: Kamran Saberifard
# Email: kamisaberi@gmail.com
# GitHub: https://github.com/kamisaberi
#
# System Dependency Installer Script
# Description:
#   Automatically detects operating system and installs required development
#   libraries for various multimedia and machine learning dependencies.
# Supported Systems:
#   - Ubuntu, Linux Mint (Debian-based)
#   - Arch Linux, Manjaro (Arch-based)
#   - macOS (Homebrew)
#   - Windows (Chocolatey, vcpkg for C++ libraries)
###############################################################################

# -----------------------------------------------------------------------------
# System Detection
# -----------------------------------------------------------------------------
# Detect operating system by checking uname or /etc/os-release
# Sets SYSTEM variable to: ubuntu, mint, arch, manjaro, macos, or windows
# Exits with error if unsupported system is detected
# -----------------------------------------------------------------------------
if [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    SYSTEM="windows"
elif grep -q "ID=ubuntu" /etc/os-release 2>/dev/null; then
    SYSTEM="ubuntu"
elif grep -q "ID=arch" /etc/os-release 2>/dev/null; then
    SYSTEM="arch"
elif grep -q "ID=manjaro" /etc/os-release 2>/dev/null; then
    SYSTEM="manjaro"
elif grep -q "ID=linuxmint" /etc/os-release 2>/dev/null; then
    SYSTEM="mint"
else
    echo "Unsupported system"
    exit 1
fi

echo "Detected system: $SYSTEM"

# -----------------------------------------------------------------------------
# Package Installation Function
# -----------------------------------------------------------------------------
# Parameters:
#   $1 - Package name (system-specific)
#   $2 - Installation command (for logging purposes)
# Functionality:
#   - Checks if package is already installed
#   - Installs package using appropriate package manager
#   - Supports apt (Debian), pacman (Arch), brew (macOS), choco (Windows)
#   - Skips installation if package is already present
# -----------------------------------------------------------------------------
check_and_install() {
    local package=$1
    local install_cmd=$2

    case "$SYSTEM" in
        ubuntu|mint)
            if ! dpkg -s "$package" >/dev/null 2>&1; then
                echo "Package $package not found, installing..."
                sudo apt-get install -y "$package"
            else
                echo "Package $package is already installed"
            fi
            ;;
        arch|manjaro)
            if ! pacman -Q "$package" >/dev/null 2>&1; then
                echo "Package $package not found, installing..."
                sudo pacman -S --noconfirm "$package"
            else
                echo "Package $package is already installed"
            fi
            ;;
        macos)
            if ! brew list "$package" >/dev/null 2>&1; then
                echo "Package $package not found, installing..."
                brew install "$package"
            else
                echo "Package $package is already installed"
            fi
            ;;
        windows)
            if [[ "$install_cmd" == "vcpkg install"* ]]; then
                # Check if vcpkg package is already installed
                if [[ -d "$VCPKG_ROOT/installed/x64-windows/lib" && -n "$(find "$VCPKG_ROOT/installed/x64-windows/lib" -name "*$package*")" ]]; then
                    echo "Package $package is already installed via vcpkg"
                else
                    echo "Package $package not found, installing via vcpkg..."
                    $install_cmd
                fi
            else
                # Chocolatey package check
                if ! choco list --local-only | grep -q "$package"; then
                    echo "Package $package not found, installing via Chocolatey..."
                    choco install "$package" -y
                else
                    echo "Package $package is already installed"
                fi
            fi
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Package Mapping Table
# -----------------------------------------------------------------------------
# Maps Ubuntu/Mint package names to Arch, macOS, and Windows equivalents
# Format: [Ubuntu Package]=Arch Package:macOS Package:Windows Package
# Windows packages use Chocolatey names where available, vcpkg names otherwise
# Key Dependencies:
#   - libopencv-dev: OpenCV computer vision library
#   - libhdf5-dev: HDF5 data format support (for OpenCV)
#   - libcurl4-openssl-dev: HTTP client library
#   - zlib1g-dev: Compression library
#   - libarchive-dev: Multi-format archive library
# Notes:
#   - Windows uses vcpkg for libtar, others use Chocolatey where possible
#   - vcpkg packages are prefixed with 'vcpkg:' to distinguish them
# -----------------------------------------------------------------------------
declare -A packages
packages["libcurl4-openssl-dev"]="curl:curl:curl"
packages["libopencv-dev"]="opencv:opencv:opencv"
packages["zlib1g-dev"]="zlib:zlib:zlib"
packages["libssl-dev"]="openssl:openssl:openssl"
packages["liblzma-dev"]="xz:xz:xz-utils"
packages["libarchive-dev"]="libarchive:libarchive:libarchive"
packages["libtar-dev"]="libtar:libtar:vcpkg:libtar"
packages["libzip-dev"]="libzip:libzip:libzip"
packages["libsndfile1-dev"]="libsndfile:libsndfile:libsndfile"
packages["libhdf5-dev"]="hdf5:hdf5:hdf5"

# -----------------------------------------------------------------------------
# Pre-Installation Checks
# -----------------------------------------------------------------------------
# Ensure package managers are installed and updated
# - Homebrew for macOS
# - Chocolatey and vcpkg for Windows
# - Update package lists for apt and pacman
# -----------------------------------------------------------------------------
case "$SYSTEM" in
    ubuntu|mint)
        echo "Updating apt package lists..."
        sudo apt-get update
        ;;
    arch|manjaro)
        echo "Updating pacman package lists..."
        sudo pacman -Syu --noconfirm
        ;;
    macos)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Homebrew not found. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            if ! command -v brew >/dev/null 2>&1; then
                echo "Homebrew installation failed. Please install manually: https://brew.sh"
                exit 1
            fi
            if [[ ":$PATH:" != *"/usr/local/bin:"* && ":$PATH:" != *"/opt/homebrew/bin:"* ]]; then
                echo "Adding Homebrew to PATH..."
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
                eval "$(/opt/homebrew/bin/brew shellenv)"
            fi
        fi
        echo "Updating Homebrew..."
        brew update
        ;;
    windows)
        # Check for PowerShell
        if ! command -v powershell >/dev/null 2>&1; then
            echo "PowerShell is required to install Chocolatey/vcpkg. Please install PowerShell first."
            exit 1
        fi
        # Check for Chocolatey
        if ! command -v choco >/dev/null 2>&1; then
            echo "Chocolatey not found. Installing Chocolatey..."
            powershell -NoProfile -ExecutionPolicy Bypass -Command "[System.Net.WebClient]::new().DownloadString('https://chocolatey.org/install.ps1') | iex"
            if ! command -v choco >/dev/null 2>&1; then
                echo "Chocolatey installation failed. Please install manually: https://chocolatey.org/install"
                exit 1
            fi
        fi
        echo "Refreshing Chocolatey environment..."
        refreshenv
        # Check for vcpkg
        VCPKG_ROOT="$HOME/vcpkg"
        if [[ ! -f "$VCPKG_ROOT/vcpkg" ]]; then
            echo "vcpkg not found. Installing vcpkg..."
            if ! command -v git >/dev/null 2>&1; then
                echo "Installing git via Chocolatey (required for vcpkg)..."
                choco install git -y
                refreshenv
            fi
            git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
            powershell -NoProfile -ExecutionPolicy Bypass -Command "cd $VCPKG_ROOT; .\bootstrap-vcpkg.bat"
            if [[ ! -f "$VCPKG_ROOT/vcpkg" ]]; then
                echo "vcpkg installation failed. Please install manually: https://vcpkg.io"
                exit 1
            fi
            # Run vcpkg integrate for MSVC
            powershell -NoProfile -ExecutionPolicy Bypass -Command "cd $VCPKG_ROOT; .\vcpkg integrate"
            echo "vcpkg installed. Use 'vcpkg integrate' in your project to link libraries."
        fi
        ;;
esac

# -----------------------------------------------------------------------------
# Main Installation Loop
# -----------------------------------------------------------------------------
# Iterates through package list and installs system-specific packages
# Handles Debian-based, Arch-based, macOS, and Windows differently
# Uses vcpkg for specific Windows packages (e.g., libtar)
# -----------------------------------------------------------------------------
for ubuntu_pkg in "${!packages[@]}"; do
    IFS=':' read -r arch_pkg macos_pkg windows_pkg <<< "${packages[$ubuntu_pkg]}"
    case "$SYSTEM" in
        ubuntu|mint)
            check_and_install "$ubuntu_pkg" "apt-get install -y"
            ;;
        arch|manjaro)
            check_and_install "$arch_pkg" "pacman -S --noconfirm"
            ;;
        macos)
            check_and_install "$macos_pkg" "brew install"
            ;;
        windows)
            if [[ "$windows_pkg" == vcpkg:* ]]; then
                vcpkg_pkg="${windows_pkg#vcpkg:}"
                check_and_install "$vcpkg_pkg" "powershell -NoProfile -ExecutionPolicy Bypass -Command \"cd $VCPKG_ROOT; .\vcpkg install $vcpkg_pkg:x64-windows\""
            else
                check_and_install "$windows_pkg" "choco install -y"
            fi
            ;;
    esac
done

# -----------------------------------------------------------------------------
# LibTorch Special Handling
# -----------------------------------------------------------------------------
# LibTorch cannot be installed via system package managers
# Provides manual installation instructions for all systems
# -----------------------------------------------------------------------------
echo "LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/"
