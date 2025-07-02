#!/bin/bash
# This script is executed by 'make install'.
# It receives the installation prefix (e.g., /usr/local) as its first argument.

set -e # Exit immediately if a command fails

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: Installation prefix not provided to setup.sh." >&2
    exit 1
fi

# Capture the installation prefix from the first command-line argument
INSTALL_PREFIX="$1"
PROJECT_NAME_LOWER="xtorch" # You can hardcode this or pass as a second argument

# Define the directory for the virtual environment
VENV_DIR="${INSTALL_PREFIX}/share/${PROJECT_NAME_LOWER}/venv"

echo "--- [xTorch] Running setup.sh ---"
echo "-> Install Prefix: ${INSTALL_PREFIX}"
echo "-> Creating venv at: ${VENV_DIR}"

# Create the venv using the system's python3
# This relies on 'python3' being available in the system's PATH
python3 -m venv "${VENV_DIR}"

# Install requirements
# This assumes a requirements.txt file was already copied to this location
REQUIREMENTS_FILE="${INSTALL_PREFIX}/share/${PROJECT_NAME_LOWER}/requirements.txt"
if [ -f "${REQUIREMENTS_FILE}" ]; then
    echo "-> Installing dependencies from ${REQUIREMENTS_FILE}"
    "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
else
    echo "-> Warning: requirements.txt not found. Skipping dependency installation."
fi

echo "--- [xTorch] setup.sh complete. ---"