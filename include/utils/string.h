/**
 * @file string.h
 * @brief String utility functions for splitting and trimming strings
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This header provides utility functions for string manipulation operations,
 * including splitting strings by delimiters and trimming whitespace.
 */

#pragma once

// Standard library includes
#include <iostream>    // For input/output operations
//#include <string>    // (Commented out) For string operations
#include <sstream>     // For string stream operations
#include <vector>      // For vector container

// Using standard namespace (note: generally avoided in header files)
using namespace std;

/**
 * @namespace xt::utils::string
 * @brief Namespace for string utility functions
 *
 * Contains utility functions for common string manipulation tasks.
 */
namespace xt::utils::string {

    /**
     * @brief Splits a string into substrings using a delimiter
     * @param str The input string to split
     * @param delim The delimiter string to split by
     * @return vector<std::string> A vector containing the split substrings
     *
     * This function:
     * - Splits the input string at each occurrence of the delimiter
     * - Returns a vector of the substrings between the delimiters
     * - Handles consecutive delimiters (empty strings between delimiters are included)
     * - Is case-sensitive
     *
     * Example usage:
     * @code
     * vector<string> parts = xt::utils::string::split("one,two,three", ",");
     * // Returns {"one", "two", "three"}
     * @endcode
     */
    vector<std::string> split(const std::string& str, const std::string& delim);

    /**
     * @brief Trims whitespace from both ends of a string
     * @param str The string to trim (modified in-place)
     * @return std::string Reference to the trimmed string
     *
     * This function:
     * - Removes all whitespace characters from the beginning and end
     * - Modifies the input string directly (in-place operation)
     * - Returns a reference to the modified string for method chaining
     * - Handles all standard whitespace characters (space, tab, newline, etc.)
     *
     * Example usage:
     * @code
     * string s = "   hello world  \t\n";
     * xt::utils::string::trim(s);
     * // s becomes "hello world"
     * @endcode
     */
    std::string trim(std::string& str);

} // namespace xt::utils::string

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