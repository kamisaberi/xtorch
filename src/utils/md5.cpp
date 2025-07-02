/**
 * @file md5.cpp
 * @brief MD5 checksum calculation implementation
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements MD5 checksum calculation using OpenSSL's EVP interface.
 * Provides a function to compute the MD5 hash of a file's contents.
 */

#include "../../include/utils/md5.h"

namespace xt::utils {

    /**
     * @brief Calculates the MD5 checksum of a file
     * @param filename Path to the file to calculate checksum for
     * @return std::string Hexadecimal string representation of the MD5 checksum (32 characters)
     *         Returns empty string on failure
     *
     * This function:
     * - Opens the file in binary mode
     * - Reads the file in 1KB chunks
     * - Computes MD5 hash using OpenSSL's EVP interface
     * - Returns the hash as a lowercase hexadecimal string
     *
     * Error handling:
     * - Returns empty string if file cannot be opened
     * - Returns empty string if MD5 computation fails
     * - Prints OpenSSL errors to stderr on failure
     *
     * Example usage:
     * @code
     * std::string checksum = xt::utils::get_md5_checksum("file.txt");
     * if (!checksum.empty()) {
     *     std::cout << "MD5: " << checksum << std::endl;
     * }
     * @endcode
     */
    std::string get_md5_checksum(const std::string &filename) {
        unsigned char mdValue[EVP_MAX_MD_SIZE];
        unsigned int mdLength;

        // Initialize OpenSSL MD5 context
        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) {
            std::cerr << "Failed to create MD_CTX." << std::endl;
            return "";
        }

        // Set up MD5 digest
        if (EVP_DigestInit_ex(mdctx, EVP_md5(), nullptr) != 1) {
            ERR_print_errors_fp(stderr);
            EVP_MD_CTX_free(mdctx);
            return "";
        }

        // Open file in binary mode
        std::ifstream file(filename, std::ifstream::binary);
        if (!file) {
            std::cerr << "Could not open file: " << filename << std::endl;
            EVP_MD_CTX_free(mdctx);
            return "";
        }

        // Process file in chunks
        char buffer[1024];
        while (file.read(buffer, sizeof(buffer))) {
            EVP_DigestUpdate(mdctx, buffer, file.gcount());
        }
        // Process remaining data
        EVP_DigestUpdate(mdctx, buffer, file.gcount());

        // Finalize digest
        if (EVP_DigestFinal_ex(mdctx, mdValue, &mdLength) != 1) {
            ERR_print_errors_fp(stderr);
            EVP_MD_CTX_free(mdctx);
            return "";
        }

        // Clean up context
        EVP_MD_CTX_free(mdctx);

        // Convert to hexadecimal string
        std::ostringstream result;
        for (unsigned int i = 0; i < mdLength; ++i) {
            result << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<int>(mdValue[i]);
        }
        return result.str();
    }

    /*
     * Alternative implementation using legacy MD5 interface (commented out)
     * This shows the older approach before OpenSSL's EVP interface was recommended
     * Kept for reference and educational purposes
     */
    //
    //std::string get_md5_checksum(const std::string &filename) {
    //    unsigned char c[MD5_DIGEST_LENGTH];
    //    MD5_CTX mdContext;
    //    MD5_Init(&mdContext);
    //
    //    std::ifstream file(filename, std::ifstream::binary);
    //    if (!file) {
    //        std::cerr << "Could not open file: " << filename << std::endl;
    //        return "";
    //    }
    //
    //    char buffer[1024];
    //    while (file.read(buffer, sizeof(buffer))) {
    //        MD5_Update(&mdContext, buffer, file.gcount());
    //    }
    //    MD5_Update(&mdContext, buffer, file.gcount());
    //    MD5_Final(c, &mdContext);
    //
    //    std::ostringstream result;
    //    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
    //        result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c[i]);
    //    }
    //    return result.str();
    //}

} // namespace xt::utils

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
 * Dependencies:
 * - OpenSSL (libcrypto) for MD5 computation
 * - C++ Standard Library for file I/O and string handling
 *
 * Security Notes:
 * - MD5 is considered cryptographically broken for security purposes
 * - Suitable only for non-cryptographic uses like file integrity checking
 */