/**
 * @file extract.h
 * @brief Archive extraction utilities for various compression formats
 *
 * Provides comprehensive file extraction capabilities for multiple archive formats
 * including XZ, GZIP, TAR, ZIP, and TGZ. Uses libarchive, zlib, libzip, and other
 * system libraries for cross-platform archive handling.
 *
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @version 1.0
 */

#pragma once

#include <iostream>
#include <zip.h>
#include <filesystem>
#include <zlib.h>
#include <fstream>
#include <tar.h>
#include <archive.h>
#include <archive_entry.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libtar.h>
#include <fcntl.h>
#include <lzma.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <archive.h>
#include <archive_entry.h>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

namespace xt::utils {
    /**
     * @brief Extracts XZ compressed file
     * @param inputFile Path to the .xz input file
     * @param outputFile Path for the decompressed output file
     * @throws runtime_error If decompression fails
     * @note Uses LZMA (XZ) compression library
     * @warning Overwrites output file if it exists
     */
    void extractXZ(const std::string &inputFile, const std::string &outputFile);

    /**
     * @brief Extracts GZIP compressed file
     * @param inFile Path to the .gz input file
     * @param outFile Optional output path (defaults to input file without .gz extension)
     * @return tuple<bool, string>
     *         - bool: true if extraction succeeded
     *         - string: output path or error message
     * @note Uses zlib for decompression
     */
    std::tuple<bool, string> extractGzip(const std::string &inFile, const std::string &outFile = "");

    /**
     * @brief Extracts TAR archive
     * @param tarFile Path to the .tar file
     * @param outPath Output directory (defaults to current directory)
     * @return true if extraction succeeded
     * @note Uses libarchive for portable tar handling
     * @warning Preserves file permissions from archive
     */
    bool extractTar(const std::string &tarFile, const std::string &outPath = "./");

    /**
     * @brief Extracts ZIP archive
     * @param inFile Path to the .zip file
     * @param outPath Output directory (defaults to current directory)
     * @return true if extraction succeeded
     * @note Uses libzip library
     * @warning Overwrites existing files in output directory
     */
    bool extractZip(const std::string &inFile, const std::string &outPath = "./");

    /**
     * @brief Extracts TGZ (tar.gz) archive
     * @param inFile Path to the .tgz or .tar.gz file
     * @param outPath Output directory (defaults to current directory)
     * @return true if extraction succeeded
     * @note Combines gzip decompression and tar extraction
     */
    bool extractTgz(const std::string &inFile, const std::string &outPath = "./");

    /**
     * @brief Auto-detects and extracts any supported archive format
     * @param inFile Input archive file path
     * @param outFile Output path (optional, format-dependent)
     * @return true if extraction succeeded
     * @throws runtime_error If format is not supported
     * @note Supports: .xz, .gz, .tar, .zip, .tgz, .tar.gz
     * @warning May create directories if outPath doesn't exist
     */
    bool extract(const std::string &inFile, const std::string &outFile = "");
}