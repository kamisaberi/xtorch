#!/bin/bash
#
# This script is a template used by CMake to generate the final script
# that runs during 'make install'. Its purpose is to create a Python
# virtual environment (venv) and install all required packages for
# the xtorch library.
#
# Using a self-contained venv makes the library relocatable and prevents
# conflicts with the system's Python packages.
#

# --- Script Configuration ---
# Exit immediately if any command fails. This prevents a partially
# completed installation.
set -e

# --- Variables Replaced by CMake at Build Time ---
# The '@VAR@' syntax is a placeholder that CMake's 'configure_file' command
# will replace with the corresponding CMake variable's value.

# The root directory where the library is being installed (e.g., /usr/local).
# This is set by the user via `cmake -DCMAKE_INSTALL_PREFIX=...`
CMAKE_INSTALL_PREFIX="@CMAKE_INSTALL_PREFIX@"

# The full path to the Python 3 interpreter found by CMake on the build machine.
# This interpreter will be used to create the virtual environment.
PYTHON_EXECUTABLE="@Python3_EXECUTABLE@"

# The full path to the requirements.txt file *after* it has been installed.
# We use the installed path to ensure the script works correctly even if the
# build directory is removed.
INSTALLED_REQUIREMENTS_PATH="@INSTALLED_REQUIREMENTS_PATH@"
# --- End of CMake Variables ---


# --- Script Logic ---
# Define the final target directory for the virtual environment. This path is
# relative to the installation prefix, making the entire installation relocatable.
# Structure: <prefix>/share/xtorch/venv
VENV_DIR="${CMAKE_INSTALL_PREFIX}/share/xtorch/venv"

echo "================================================="
echo "=== xtorch Post-Install: Setting up Python venv ==="
echo "================================================="
echo "-> Installation Prefix: ${CMAKE_INSTALL_PREFIX}"
echo "-> Venv Directory:      ${VENV_DIR}"
echo "-> Requirements File:   ${INSTALLED_REQUIREMENTS_PATH}"
echo "-> Python Used:         ${PYTHON_EXECUTABLE}"
echo "-------------------------------------------------"

# Validation: Ensure the Python executable was found by CMake.
if [ -z "${PYTHON_EXECUTABLE}" ]; then
    echo "[ERROR] Python executable not found by CMake. Cannot create venv." >&2
    echo "[ERROR] Please ensure Python 3 is installed and in your PATH." >&2
    exit 1
fi

# Validation: Ensure the requirements file exists at the expected location.
if [ ! -f "${INSTALLED_REQUIREMENTS_PATH}" ]; then
    echo "[ERROR] requirements.txt not found at installed location: ${INSTALLED_REQUIREMENTS_PATH}" >&2
    exit 1
fi

# Step 1: Create the Virtual Environment
# Check if the venv directory already exists. This makes the installation idempotent
# (i.e., running 'make install' multiple times won't cause errors).
if [ -d "${VENV_DIR}" ]; then
    echo "-> Virtual environment already exists. Skipping creation."
else
    echo "-> Creating virtual environment..."
    # Use the Python interpreter found by CMake to create the venv.
    # The '--clear' option can be added if you want to ensure it's a fresh venv,
    # but it's safer to just check for existence.
    "${PYTHON_EXECUTABLE}" -m venv "${VENV_DIR}"
    echo "-> Virtual environment created successfully."
fi

# Define paths to the Python and pip executables inside our new venv.
# This is more robust than relying on 'source activate'.
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Step 2: Upgrade Pip
# It's good practice to ensure pip is up-to-date within the venv.
echo "-> Upgrading pip inside the venv..."
"${VENV_PYTHON}" -m pip install --upgrade pip

# Step 3: Install Dependencies
# Use the venv's pip to install all packages listed in requirements.txt.
echo "-> Installing Python dependencies from requirements.txt..."
"${VENV_PYTHON}" -m pip install -r "${INSTALLED_REQUIREMENTS_PATH}"

echo "-------------------------------------------------"
echo "=== Python venv setup for xtorch complete. ==="
echo "================================================="

# Exit with success status
exit 0