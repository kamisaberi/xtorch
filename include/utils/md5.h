/**
 * @file md5.h
 * @brief MD5 checksum utility functions
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This header provides utility functions for calculating MD5 checksums of files.
 * Uses OpenSSL's EVP interface for cryptographic operations.
 */

#pragma once

// Standard library includes
#include <iostream>   // For input/output operations
#include <fstream>    // For file operations
#include <iomanip>    // For output formatting

// OpenSSL includes
#include <openssl/evp.h>  // For EVP (Envelope) digest functions
#include <openssl/err.h>  // For OpenSSL error handling

/**
 * @namespace xt::utils
 * @brief Namespace for utility functions
 *
 * Contains various utility functions including cryptographic operations.
 */
namespace xt::utils {

    /**
     * @brief Calculates MD5 checksum of a file
     * @param filename Path to the file to calculate checksum for
     * @return std::string Hexadecimal string representation of the MD5 checksum
     * @throws std::runtime_error if file cannot be opened or checksum calculation fails
     *
     * This function:
     * - Opens the file in binary mode
     * - Reads the file in chunks
     * - Calculates MD5 hash using OpenSSL's EVP interface
     * - Returns the hash as a 32-character hexadecimal string
     *
     * Example usage:
     * @code
     * std::string checksum = xt::utils::get_md5_checksum("file.txt");
     * @endcode
     *
     * Note: Requires OpenSSL library to be linked.
     */
    std::string get_md5_checksum(const std::string &filename);

} // namespace xt::utils

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Utility Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * Licensed under the MIT License. See LICENSE file in the project root for full license information.
 *
 * OpenSSL License Notice:
 * This product includes software developed by the OpenSSL Project
 * for use in the OpenSSL Toolkit (https://www.openssl.org/)
 */