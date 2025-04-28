/**
 * @file string.cpp
 * @brief String manipulation utilities implementation
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements common string manipulation utilities including:
 * - String splitting by delimiter
 * - Whitespace trimming
 */

#include "../../include/utils/string.h"

namespace xt::utils::string {

    /**
     * @brief Splits a string into tokens using a delimiter
     * @param str The input string to split
     * @param delim The delimiter string to split on
     * @return vector<std::string> List of tokens (non-empty strings between delimiters)
     *
     * This function:
     * - Handles arbitrary length delimiters
     * - Skips empty tokens between consecutive delimiters
     * - Returns all non-empty substrings between delimiters
     * - Preserves original string content (no modifications)
     *
     * Example usage:
     * @code
     * auto parts = xt::utils::string::split("one,two,three", ",");
     * // Returns {"one", "two", "three"}
     *
     * auto words = xt::utils::string::split("hello  world   c++", " ");
     * // Returns {"hello", "world", "c++"} (skips empty spaces)
     * @endcode
     */
    vector<std::string> split(const std::string& str, const std::string& delim) {
        vector<std::string> tokens;
        size_t prev = 0, pos = 0;
        do {
            // Find next delimiter occurrence
            pos = str.find(delim, prev);
            if (pos == std::string::npos) pos = str.length();

            // Extract token between previous and current position
            std::string token = str.substr(prev, pos-prev);

            // Add non-empty tokens to results
            if (!token.empty()) tokens.push_back(token);

            // Move past current delimiter
            prev = pos + delim.length();
        }
        while (pos < str.length() && prev < str.length());

        return tokens;
    }

    /**
     * @brief Trims leading and trailing whitespace from a string
     * @param str The string to trim (modified in-place)
     * @return std::string Reference to the trimmed string
     *
     * This function:
     * - Removes all leading and trailing space characters (' ')
     * - Modifies the input string directly
     * - Returns reference to modified string for method chaining
     * - Handles all-whitespace strings (result will be empty)
     *
     * Example usage:
     * @code
     * std::string s = "   hello world   ";
     * xt::utils::string::trim(s);
     * // s now contains "hello world"
     *
     * // Method chaining example:
     * auto trimmed = xt::utils::string::trim(s).substr(0, 5);
     * @endcode
     */
    std::string trim(std::string& str) {
        // Remove trailing spaces
        str.erase(str.find_last_not_of(' ')+1);

        // Remove leading spaces
        str.erase(0, str.find_first_not_of(' '));

        return str;
    }

} // namespace xt::utils::string

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Utility Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * License: MIT
 *
 * Notes:
 * - For more advanced string splitting/trimming needs, consider using
 *   regular expressions or boost::algorithm
 * - These implementations focus on simplicity and common use cases
 */