#!/bin/bash

PACKAGE_NAME="libtar"

if pacman -Q "${PACKAGE_NAME}" &> /dev/null; then
    echo "${PACKAGE_NAME} is installed."
else
    echo "${PACKAGE_NAME} is not installed."
fi

PACKAGE_NAME="libzip"

if pacman -Q "${PACKAGE_NAME}" &> /dev/null; then
    echo "${PACKAGE_NAME} is installed."
else
    echo "${PACKAGE_NAME} is not installed."
fi

PACKAGE_NAME="xz"

if pacman -Q "${PACKAGE_NAME}" &> /dev/null; then
    echo "${PACKAGE_NAME} is installed."
else
    echo "${PACKAGE_NAME} is not installed."
fi