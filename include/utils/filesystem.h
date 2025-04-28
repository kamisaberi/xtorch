/**
 * @file filesystem_utils.h
 * @brief Filesystem utility functions for counting files
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This header provides utility functions for filesystem operations,
 * particularly for counting files in directories.
 */

#pragma once

// Standard library includes
#include <filesystem>  // For filesystem operations
#include <algorithm>   // For algorithmic operations
#include <iostream>    // For input/output operations

// Alias namespace for cleaner code
namespace fs = std::filesystem;

/**
 * @namespace xt::utils::fs
 * @brief Namespace for filesystem utility functions
 *
 * Contains utility functions that extend standard filesystem functionality.
 */
namespace xt::utils::fs {

    /**
     * @brief Counts files in a directory
     * @param path The filesystem path to count files in
     * @param recursive Whether to count files recursively in subdirectories (default: true)
     * @return std::size_t The number of files found
     *
     * This function counts all regular files in the specified directory.
     * When recursive is true, it will also count files in all subdirectories.
     *
     * Example usage:
     * @code
     * size_t file_count = xt::utils::fs::countFiles("/path/to/directory");
     * @endcode
     */
    std::size_t countFiles(const std::filesystem::path& path, bool recursive = true);

} // namespace xt::utils::fs

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Utility Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * Licensed under the MIT License. See LICENSE file in the project root for full license information.
 */