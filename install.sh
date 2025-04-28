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
#   Prompts for LibTorch path (with validation loop), updates CMakeLists.txt, and runs make install.
# Supported Systems:
#   - Ubuntu, Linux Mint (Debian-based)
#   - Arch Linux, Manjaro (Arch-based)
#   - Fedora (dnf-based)
#   - openSUSE (zypper-based)
#   - macOS (Homebrew)
#   - Windows (Chocolatey, vcpkg for C++ libraries)
###############################################################################

# -----------------------------------------------------------------------------
# System Detection
# -----------------------------------------------------------------------------
# Detect operating system by checking uname or /etc/os-release
# Sets SYSTEM variable to: ubuntu, mint, arch, manjaro, fedora, suse, macos, or windows
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
elif grep -q "ID=fedora" /etc/os-release 2>/dev/null; then
    SYSTEM="fedora"
elif grep -q "ID=opensuse" /etc/os-release 2>/dev/null || grep -q "ID=suse" /etc/os-release 2>/dev/null; then
    SYSTEM="suse"
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
#   - Supports apt (Debian), pacman (Arch), dnf (Fedora), zypper (SUSE), brew (macOS), choco/vcpkg (Windows)
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
        fedora)
            if ! rpm -q "$package" >/dev/null 2>&1; then
                echo "Package $package not found, installing..."
                sudo dnf install -y "$package"
            else
                echo "Package $package is already installed"
            fi
            ;;
        suse)
            if ! zypper se -i "$package" >/dev/null 2>&1; then
                echo "Package $package not found, installing..."
                sudo zypper install -y "$package"
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
                if [[ -d "$VCPKG_ROOT/installed/x64-windows/lib" && -n "$(find "$VCPKG_ROOT/installed/x64-windows/lib" -name "*$package*")" ]]; then
                    echo "Package $package is already installed via vcpkg"
                else
                    echo "Package $package not found, installing via vcpkg..."
                    $install_cmd
                fi
            else
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
# Maps Ubuntu/Mint package names to Arch, Fedora, SUSE, macOS, and Windows equivalents
# Format: [Ubuntu Package]=Arch Package:Fedora Package:SUSE Package:macOS Package:Windows Package
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
#   - Fedora and SUSE use -devel suffix for development packages
# -----------------------------------------------------------------------------
declare -A packages
packages["libcurl4-openssl-dev"]="curl:libcurl-devel:libcurl-devel:curl:curl"
packages["libopencv-dev"]="opencv:opencv-devel:opencv-devel:opencv:opencv"
packages["zlib1g-dev"]="zlib:zlib-devel:zlib-devel:zlib:zlib"
packages["libssl-dev"]="openssl:openssl-devel:libopenssl-devel:openssl:openssl"
packages["liblzma-dev"]="xz:xz-devel:xz-devel:xz:xz-utils"
packages["libarchive-dev"]="libarchive:libarchive-devel:libarchive-devel:libarchive:libarchive"
packages["libtar-dev"]="libtar:libtar-devel:libtar-devel:libtar:vcpkg:libtar"
packages["libzip-dev"]="libzip:libzip-devel:libzip-devel:libzip:libzip"
packages["libsndfile1-dev"]="libsndfile:libsndfile-devel:libsndfile-devel:libsndfile:libsndfile"
packages["libhdf5-dev"]="hdf5:hdf5-devel:hdf5-devel:hdf5:hdf5"

# -----------------------------------------------------------------------------
# Pre-Installation Checks
# -----------------------------------------------------------------------------
# Ensure package managers are installed and updated
# - Homebrew for macOS
# - Chocolatey and vcpkg for Windows
# - Update package lists for apt, pacman, dnf, zypper
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
    fedora)
        echo "Updating dnf package lists..."
        sudo dnf update -y
        ;;
    suse)
        echo "Updating zypper package lists..."
        sudo zypper refresh
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
        if ! command -v powershell >/dev/null 2>&1; then
            echo "PowerShell is required to install Chocolatey/vcpkg. Please install PowerShell first."
            exit 1
        fi
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
            powershell -NoProfile -ExecutionPolicy Bypass -Command "cd $VCPKG_ROOT; .\vcpkg integrate"
            echo "vcpkg installed. Use 'vcpkg integrate' in your project to link libraries."
        fi
        ;;
esac

# -----------------------------------------------------------------------------
# Main Installation Loop
# -----------------------------------------------------------------------------
# Iterates through package list and installs system-specific packages
# Handles Debian-based, Arch-based, Fedora, SUSE, macOS, and Windows differently
# Uses vcpkg for specific Windows packages (e.g., libtar)
# -----------------------------------------------------------------------------
for ubuntu_pkg in "${!packages[@]}"; do
    IFS=':' read -r arch_pkg fedora_pkg suse_pkg macos_pkg windows_pkg <<< "${packages[$ubuntu_pkg]}"
    case "$SYSTEM" in
        ubuntu|mint)
            check_and_install "$ubuntu_pkg" "apt-get install -y"
            ;;
        arch|manjaro)
            check_and_install "$arch_pkg" "pacman -S --noconfirm"
            ;;
        fedora)
            check_and_install "$fedora_pkg" "dnf install -y"
            ;;
        suse)
            check_and_install "$suse_pkg" "zypper install -y"
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
# LibTorch Installation and CMakeLists.txt Update
# -----------------------------------------------------------------------------
# Prompt user for LibTorch path until valid, update CMakeLists.txt, and run make install
# - Loops to prompt for absolute path to LibTorch until valid
# - Validates path is absolute and LibTorch exists
# - Updates CMakeLists.txt with the provided path
# - Runs sudo make install (Linux/macOS) or equivalent on Windows
# -----------------------------------------------------------------------------
echo "LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/"

# Loop to prompt for LibTorch path until valid
while true; do
    read -p "Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): " LIBTORCH_PATH

    # Allow user to quit
    if [[ "$LIBTORCH_PATH" == "quit" || "$LIBTORCH_PATH" == "exit" ]]; then
        echo "Exiting due to user request."
        exit 1
    fi

    # Validate absolute path
    case "$SYSTEM" in
        ubuntu|mint|arch|manjaro|fedora|suse|macos)
            if [[ ! "$LIBTORCH_PATH" =~ ^/ ]]; then
                echo "Error: Path must be absolute (start with '/'). Please try again."
                continue
            fi
            ;;
        windows)
            if [[ ! "$LIBTORCH_PATH" =~ ^[a-zA-Z]:\\ ]]; then
                echo "Error: Path must be absolute (start with drive letter, e.g., C:\\). Please try again."
                continue
            fi
            # Convert Windows path to CMake-compatible format (forward slashes)
            LIBTORCH_PATH=$(echo "$LIBTORCH_PATH" | sed 's|\\|/|g')
            ;;
    esac

    # Validate LibTorch directory exists
    case "$SYSTEM" in
        ubuntu|mint|arch|manjaro|fedora|suse)
            if [[ ! -f "$LIBTORCH_PATH/lib/libtorch.so" ]]; then
                echo "Error: LibTorch not found at $LIBTORCH_PATH. Ensure the path contains a valid LibTorch installation with 'lib/libtorch.so'."
                continue
            fi
            ;;
        macos)
            if [[ ! -f "$LIBTORCH_PATH/lib/libtorch.dylib" ]]; then
                echo "Error: LibTorch not found at $LIBTORCH_PATH. Ensure the path contains a valid LibTorch installation with 'lib/libtorch.dylib'."
                continue
            fi
            ;;
        windows)
            if [[ ! -f "$LIBTORCH_PATH/lib/torch.lib" ]]; then
                echo "Error: LibTorch not found at $LIBTORCH_PATH. Ensure the path contains a valid LibTorch installation with 'lib/torch.lib'."
                continue
            fi
            ;;
    esac

    # If we reach here, the path is valid
    echo "Valid LibTorch path provided: $LIBTORCH_PATH"
    break
done

# Locate and update CMakeLists.txt
CMAKELISTS_FILE="CMakeLists.txt"
if [[ ! -f "$CMAKELISTS_FILE" ]]; then
    echo "Error: CMakeLists.txt not found in the current directory."
    read -p "Enter the path to CMakeLists.txt: " CMAKELISTS_FILE
    if [[ ! -f "$CMAKELISTS_FILE" ]]; then
        echo "Error: Specified CMakeLists.txt does not exist."
        exit 1
    fi
fi

# Backup CMakeLists.txt
cp "$CMAKELISTS_FILE" "${CMAKELISTS_FILE}.bak"
echo "Backed up $CMAKELISTS_FILE to ${CMAKELISTS_FILE}.bak"

# Update CMakeLists.txt
if grep -q "list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)" "$CMAKELISTS_FILE"; then
    sed -i "s|list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)|list(APPEND CMAKE_PREFIX_PATH $LIBTORCH_PATH)|" "$CMAKELISTS_FILE"
    echo "Updated CMakeLists.txt with LibTorch path: $LIBTORCH_PATH"
else
    echo "Warning: Could not find 'list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)' in $CMAKELISTS_FILE."
    echo "Please manually add 'list(APPEND CMAKE_PREFIX_PATH $LIBTORCH_PATH)' to $CMAKELISTS_FILE."
fi

# Run make install (Linux/macOS) or build on Windows
case "$SYSTEM" in
    ubuntu|mint|arch|manjaro|fedora|suse)
        echo "Running sudo make install..."
        sudo make install
        if [[ $? -ne 0 ]]; then
            echo "Error: 'sudo make install' failed."
            exit 1
        fi
        ;;
    macos)
        echo "Running make install..."
        make install
        if [[ $? -ne 0 ]]; then
            echo "Error: 'make install' failed."
            exit 1
        fi
        ;;
    windows)
        echo "Windows detected. Running CMake build and install..."
        if [[ -d "build" ]]; then
            rm -rf build
        fi
        mkdir build && cd build
        cmake ..
        cmake --build . --config Release
        cmake --install .
        if [[ $? -ne 0 ]]; then
            echo "Error: CMake build/install failed."
            exit 1
        fi
        cd ..
        ;;
esac

echo "Installation and configuration completed successfully."