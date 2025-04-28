/**
 * @file filesystem.h
 * @brief Filesystem utility functions
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file provides utility functions for filesystem operations,
 * including counting files in directories with optional recursion.
 */

#include "../../include/utils/filesystem.h"

namespace xt::utils::fs {

    /**
     * @brief Counts the number of regular files in a directory
     * @param path The filesystem path to count files in
     * @param recursive Whether to count files recursively in subdirectories (default: true)
     * @return std::size_t The number of regular files found
     *
     * This function counts all regular files (non-directories, non-special files) in:
     * - Only the specified directory when recursive=false
     * - The directory and all subdirectories when recursive=true
     *
     * Usage examples:
     * @code
     * // Count files in current directory only
     * size_t count = xt::utils::fs::countFiles("./", false);
     *
     * // Count files recursively in directory tree
     * size_t total = xt::utils::fs::countFiles("/path/to/dir", true);
     * @endcode
     *
     * @note Uses std::filesystem::directory_iterator for non-recursive count
     * @note Uses std::filesystem::recursive_directory_iterator for recursive count
     * @note Returns 0 if path doesn't exist or isn't a directory
     * @note Symlinks are not followed - counts the links themselves, not their targets
     */
    std::size_t countFiles(const std::filesystem::path& path, bool recursive) {
        if (recursive) {
            // Recursive count using recursive_directory_iterator
            return std::count_if(
                    std::filesystem::recursive_directory_iterator(path),
                    std::filesystem::recursive_directory_iterator{},
                    [](const std::filesystem::directory_entry& entry) {
                        return entry.is_regular_file();
                    }
            );
        } else {
            // Non-recursive count using directory_iterator
            return std::count_if(
                    std::filesystem::directory_iterator(path),
                    std::filesystem::directory_iterator{},
                    [](const std::filesystem::directory_entry& entry) {
                        return entry.is_regular_file();
                    }
            );
        }
    }

} // namespace xt::utils::fs

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Utility Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * Licensed under the MIT License. See LICENSE file in the project root
 * for full license information.
 *
 * Requirements:
 * - C++17 or later (for std::filesystem support)
 */