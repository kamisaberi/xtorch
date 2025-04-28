# System Dependency Installer Script Documentation

## Overview
**Author**: Kamran Saberifard  
**Email**: kamisaberi@gmail.com  
**GitHub**: [https://github.com/kamisaberi](https://github.com/kamisaberi)  
**Script Name**: `install_dependencies.sh`  
**Purpose**: This bash script automates the installation of development libraries required for multimedia and machine learning projects across multiple operating systems. It detects the OS, installs dependencies using appropriate package managers, prompts for a LibTorch installation path, updates a `CMakeLists.txt` file, and runs the build/install process.

### Supported Systems
- **Linux**:
  - Ubuntu, Linux Mint (Debian-based, uses `apt`)
  - Arch Linux, Manjaro (Arch-based, uses `pacman`)
- **macOS**: Uses Homebrew (`brew`)
- **Windows**: Uses Chocolatey (`choco`) and vcpkg for C++ libraries

### Key Features
- **System Detection**: Identifies the operating system and configures package installation accordingly.
- **Dependency Installation**: Installs libraries like OpenCV, HDF5, libcurl, zlib, etc., using platform-specific package managers.
- **LibTorch Configuration**: Prompts the user for a valid LibTorch path (with validation loop), updates `CMakeLists.txt`, and runs the build/install process.
- **Error Handling**: Validates inputs, checks for package manager availability, and provides clear error messages.
- **Cross-Platform Support**: Handles platform-specific nuances (e.g., path formats, build commands).

## Usage

### Prerequisites
- **Linux**: `sudo` privileges for `apt`, `pacman`, and `make install`.
- **macOS**: Xcode Command Line Tools (for `make` and `git`).
- **Windows**: PowerShell and a C++ compiler (e.g., MSVC via Visual Studio).
- **General**: Internet access for downloading package managers and dependencies.
- **LibTorch**: A manually downloaded LibTorch installation from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
- **CMake Project**: A `CMakeLists.txt` file in the current directory or a user-specified location.

### Running the Script
1. Save the script as `install_dependencies.sh`.
2. Make it executable:
   ```bash
   chmod +x install_dependencies.sh
   ```
3. Run the script:
   ```bash
   ./install_dependencies.sh
   ```

### User Inputs
- **LibTorch Path**:
  - Prompt: `Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit):`
  - Requirements:
    - Must be absolute (starts with `/` on Linux/macOS, drive letter like `C:\` on Windows).
    - Must contain a valid LibTorch installation (checked via `lib/libtorch.so` on Linux, `lib/libtorch.dylib` on macOS, `lib/torch.lib` on Windows).
    - The script loops until a valid path is provided or the user types `quit`/`exit`.
- **CMakeLists.txt Path** (if not found in the current directory):
  - Prompt: `Enter the path to CMakeLists.txt:`
  - Must point to an existing `CMakeLists.txt` file.

### Outputs
- **Console Messages**: Logs the detected system, package installation status, LibTorch validation, `CMakeLists.txt` updates, and build/install results.
- **File Modifications**:
  - Installs dependencies using the appropriate package manager.
  - Backs up `CMakeLists.txt` to `CMakeLists.txt.bak`.
  - Updates `CMakeLists.txt` with the LibTorch path.
- **Build/Install**:
  - Linux: Runs `sudo make install`.
  - macOS: Runs `make install`.
  - Windows: Runs CMake build (`cmake --build . --config Release`) and install (`cmake --install .`).

## Script Structure and Detailed Breakdown

### 1. System Detection
**Purpose**: Identifies the operating system to configure package installation and build commands.  
**Code**:
```bash
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
```
**Functionality**:
- Uses `$OSTYPE` to detect macOS (`darwin`) and Windows (`msys`, `cygwin`, `win32`).
- Parses `/etc/os-release` for Linux distributions (Ubuntu, Arch, Manjaro, Mint).
- Sets the `SYSTEM` variable to one of: `macos`, `windows`, `ubuntu`, `arch`, `manjaro`, `mint`.
- Exits with an error if the system is unsupported.

### 2. Package Installation Function
**Purpose**: Installs dependencies using platform-specific package managers, checking if they’re already installed.  
**Code**:
```bash
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
```
**Functionality**:
- Takes two parameters: package name and installation command (for logging).
- Checks if the package is installed:
  - Linux (Debian): Uses `dpkg -s`.
  - Linux (Arch): Uses `pacman -Q`.
  - macOS: Uses `brew list`.
  - Windows (Chocolatey): Uses `choco list --local-only`.
  - Windows (vcpkg): Checks for library files in `$VCPKG_ROOT/installed/x64-windows/lib`.
- Installs the package if missing, using `sudo` where required (`apt`, `pacman`).
- Skips installation if the package is already present.

### 3. Package Mapping Table
**Purpose**: Maps Ubuntu package names to equivalents for Arch, macOS, and Windows.  
**Code**:
```bash
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
```
**Functionality**:
- Uses an associative array to map Ubuntu package names to Arch, macOS, and Windows equivalents.
- Format: `[Ubuntu Package]=Arch Package:macOS Package:Windows Package`.
- Windows uses `vcpkg:` prefix for packages installed via vcpkg (e.g., `libtar`).
- Covers key dependencies like OpenCV, HDF5, libcurl, zlib, and libarchive.

### 4. Pre-Installation Checks
**Purpose**: Ensures package managers are installed and updated before installing dependencies.  
**Code**:
```bash
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
```
**Functionality**:
- **Linux (Debian)**: Updates `apt` package lists (`sudo apt-get update`).
- **Linux (Arch)**: Updates `pacman` package lists (`sudo pacman -Syu --noconfirm`).
- **macOS**:
  - Installs Homebrew if missing using the official installation script.
  - Adds Homebrew to the PATH if not present.
  - Updates Homebrew (`brew update`).
- **Windows**:
  - Checks for PowerShell (required for Chocolatey/vcpkg).
  - Installs Chocolatey if missing using the official PowerShell script.
  - Installs vcpkg if missing by cloning its GitHub repository and running `bootstrap-vcpkg.bat`.
  - Installs `git` via Chocolatey if needed for vcpkg.
  - Runs `vcpkg integrate` for MSVC integration.
  - Refreshes the environment (`refreshenv`) after Chocolatey installations.

### 5. Main Installation Loop
**Purpose**: Installs all dependencies listed in the package mapping table.  
**Code**:
```bash
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
```
**Functionality**:
- Iterates through the `packages` array.
- Extracts platform-specific package names using `IFS=':'`.
- Calls `check_and_install` with the appropriate package name and command.
- For Windows, uses vcpkg for packages prefixed with `vcpkg:` (e.g., `libtar`), otherwise uses Chocolatey.

### 6. LibTorch Installation and CMakeLists.txt Update
**Purpose**: Handles LibTorch configuration by prompting for its path, validating it, updating `CMakeLists.txt`, and running the build/install process.  
**Code**:
```bash
echo "LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/"

while true; do
    read -p "Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): " LIBTORCH_PATH
    if [[ "$LIBTORCH_PATH" == "quit" || "$LIBTORCH_PATH" == "exit" ]]; then
        echo "Exiting due to user request."
        exit 1
    fi
    case "$SYSTEM" in
        ubuntu|mint|arch|manjaro|macos)
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
            LIBTORCH_PATH=$(echo "$LIBTORCH_PATH" | sed 's|\\|/|g')
            ;;
    esac
    case "$SYSTEM" in
        ubuntu|mint|arch|manjaro)
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
    echo "Valid LibTorch path provided: $LIBTORCH_PATH"
    break
done

CMAKELISTS_FILE="CMakeLists.txt"
if [[ ! -f "$CMAKELISTS_FILE" ]]; then
    echo "Error: CMakeLists.txt not found in the current directory."
    read -p "Enter the path to CMakeLists.txt: " CMAKELISTS_FILE
    if [[ ! -f "$CMAKELISTS_FILE" ]]; then
        echo "Error: Specified CMakeLists.txt does not exist."
        exit 1
    fi
fi

cp "$CMAKELISTS_FILE" "${CMAKELISTS_FILE}.bak"
echo "Backed up $CMAKELISTS_FILE to ${CMAKELISTS_FILE}.bak"

if grep -q "list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)" "$CMAKELISTS_FILE"; then
    sed -i "s|list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)|list(APPEND CMAKE_PREFIX_PATH $LIBTORCH_PATH)|" "$CMAKELISTS_FILE"
    echo "Updated CMakeLists.txt with LibTorch path: $LIBTORCH_PATH"
else
    echo "Warning: Could not find 'list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)' in $CMAKELISTS_FILE."
    echo "Please manually add 'list(APPEND CMAKE_PREFIX_PATH $LIBTORCH_PATH)' to $CMAKELISTS_FILE."
fi

case "$SYSTEM" in
    ubuntu|mint|arch|manjaro)
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
```
**Functionality**:
- **LibTorch Prompt**:
  - Informs the user that LibTorch must be manually downloaded.
  - Prompts for an absolute path to the LibTorch installation.
  - Allows the user to exit by typing `quit` or `exit`.
- **Path Validation Loop**:
  - Checks if the path is absolute:
    - Linux/macOS: Must start with `/`.
    - Windows: Must start with a drive letter (e.g., `C:\`).
  - Converts Windows paths to CMake-compatible format (replaces `\` with `/`).
  - Verifies the LibTorch installation by checking for:
    - Linux: `lib/libtorch.so`
    - macOS: `lib/libtorch.dylib`
    - Windows: `lib/torch.lib`
  - Loops until a valid path is provided or the user quits.
- **CMakeLists.txt Update**:
  - Checks for `CMakeLists.txt` in the current directory.
  - Prompts for its path if not found and validates the provided path.
  - Backs up `CMakeLists.txt` to `CMakeLists.txt.bak`.
  - Replaces `list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)` with `list(APPEND CMAKE_PREFIX_PATH $LIBTORCH_PATH)`.
  - Warns the user if the target line is not found, instructing them to add the path manually.
- **Build/Install**:
  - Linux: Runs `sudo make install` (requires sudo privileges).
  - macOS: Runs `make install` (no sudo needed).
  - Windows: Creates a `build` directory, runs `cmake ..`, builds with `cmake --build . --config Release`, and installs with `cmake --install .`.
  - Checks the exit status and exits with an error if the build/install fails.

## Error Handling
- **Unsupported System**: Exits if the OS is not Ubuntu, Mint, Arch, Manjaro, macOS, or Windows.
- **Package Manager Missing**:
  - Installs Homebrew, Chocolatey, or vcpkg if not found.
  - Exits if installation fails, providing manual installation URLs.
- **Invalid LibTorch Path**:
  - Loops with specific error messages for non-absolute paths or missing LibTorch libraries.
  - Allows the user to quit with `quit`/`exit`.
- **CMakeLists.txt Issues**:
  - Prompts for the file path if not found in the current directory.
  - Exits if the provided path is invalid.
  - Warns if the LibTorch path line is not found in `CMakeLists.txt`.
- **Build/Install Failures**:
  - Checks the exit status of `make install` or CMake commands.
  - Exits with an error message if the command fails.

## Platform-Specific Notes
- **Linux**:
  - Requires `sudo` for `apt`, `pacman`, and `make install`.
  - Assumes `make` and `cmake` are installed (can be added to dependencies if needed).
- **macOS**:
  - Homebrew installations may require Xcode Command Line Tools.
  - No `sudo` needed for `make install` in most cases.
- **Windows**:
  - Requires PowerShell and a C++ compiler (e.g., MSVC).
  - vcpkg is used for `libtar` and requires `git` and a compiler.
  - CMake build assumes a CMake-based project; adjust if using another build system.
  - Paths are converted to use forward slashes for CMake compatibility.

## Example Run (Linux)
```bash
$ ./install_dependencies.sh
Detected system: ubuntu
Updating apt package lists...
Package libcurl4-openssl-dev is already installed
Package libopencv-dev not found, installing...
...
LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/
Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): libtorch
Error: Path must be absolute (start with '/'). Please try again.
Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): /home/user/invalid
Error: LibTorch not found at /home/user/invalid. Ensure the path contains a valid LibTorch installation with 'lib/libtorch.so'.
Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): /home/user/libtorch
Valid LibTorch path provided: /home/user/libtorch
Backed up CMakeLists.txt to CMakeLists.txt.bak
Updated CMakeLists.txt with LibTorch path: /home/user/libtorch
Running sudo make install...
[sudo] password for user: 
...
Installation and configuration completed successfully.
```

## Example Run (Windows)
```bash
$ ./install_dependencies.sh
Detected system: windows
Refreshing Chocolatey environment...
Package curl is already installed
Package opencv not found, installing via Chocolatey...
...
LibTorch is not available via package managers. Please download and install manually from https://pytorch.org/get-started/locally/
Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): C:\invalid
Error: LibTorch not found at C:\invalid. Ensure the path contains a valid LibTorch installation with 'lib/torch.lib'.
Enter the absolute path to your LibTorch installation (e.g., /home/user/libtorch or C:\\libtorch, or type 'quit' to exit): C:\libtorch
Valid LibTorch path provided: C:/libtorch
Backed up CMakeLists.txt to CMakeLists.txt.bak
Updated CMakeLists.txt with LibTorch path: C:/libtorch
Windows detected. Running CMake build and install...
...
Installation and configuration completed successfully.
```

## Troubleshooting
- **Package Installation Fails**:
  - Ensure internet access and sufficient permissions.
  - Check package manager logs (e.g., `/var/log/apt`, Chocolatey logs).
- **LibTorch Path Invalid**:
  - Verify the LibTorch installation contains the expected library file.
  - Download a fresh copy from [https://pytorch.org/](https://pytorch.org/) if needed.
- **CMakeLists.txt Not Found**:
  - Ensure the project directory contains `CMakeLists.txt` or provide the correct path.
- **Build/Install Fails**:
  - Check for missing build tools (`make`, `cmake`, MSVC).
  - Verify the project’s CMake configuration is correct.
- **Windows CMake Issues**:
  - Ensure a C++ compiler is installed (e.g., Visual Studio Build Tools).
  - Run the script in a PowerShell-enabled environment.

## Future Improvements
- **Flexible CMakeLists.txt Matching**: Use regex to match variations of the `CMAKE_PREFIX_PATH` line.
- **LibTorch Version Check**: Validate the LibTorch version by parsing `version.txt`.
- **Custom Build Options**: Allow users to specify build configurations (e.g., Debug vs. Release on Windows).
- **Additional Dependencies**: Add checks for `cmake`, `make`, or compilers if missing.
- **Path Normalization**: Trim whitespace or normalize LibTorch paths to handle edge cases.

## License
This script is provided as-is without warranty. Users are responsible for ensuring compatibility with their systems and projects. Contact the author for contributions or issues.