/**
 * @file xt_utils_download.h
 * @brief Utility functions for downloading files, including Google Drive support
 *
 * This header provides utility functions for downloading files from URLs and Google Drive.
 * It includes functionality for handling HTTP downloads via libcurl and reconstructing
 * Google Drive download links.
 *
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: 2023-11-15
 * @lastmodified: 2023-11-15
 * @version 1.0
 */

#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <curl/curl.h>

using namespace std;
namespace fs = std::filesystem;

namespace xt::utils {
    /**
     * @brief Downloads a file from a given URL to a specified output path
     * @param url The URL of the file to download
     * @param outPath The destination path where the file should be saved
     * @return A tuple containing:
     *         - bool: true if download succeeded, false otherwise
     *         - string: Success message or error description
     * @note Uses libcurl for HTTP/HTTPS downloads
     * @warning Overwrites existing files at outPath without warning
     * @example
     * auto [success, message] = download("https://example.com/file.zip", "downloaded.zip");
     */
    std::tuple<bool, std::string> download(std::string &url, std::string outPath);

    /**
     * @brief Reconstructs a direct download link from a Google Drive shareable link
     * @param gid The Google Drive file ID or shareable URL
     * @return A reconstructed direct download URL
     * @throws None
     * @note Handles both file IDs and full Google Drive URLs
     * @example
     * string directLink = rebuild_google_drive_link("1a2b3c4d5e6f7g8h9i0j");
     */
    std::string rebuild_google_drive_link(std::string gid);

    /**
     * @brief Downloads a file from Google Drive
     * @param gid The Google Drive file ID or shareable URL
     * @param outPath The destination path where the file should be saved
     * @return A tuple containing:
     *         - bool: true if download succeeded, false otherwise
     *         - string: Success message or error description
     * @note Internally uses rebuild_google_drive_link() and download()
     * @warning Google Drive has download quotas that may affect large files
     * @example
     * auto [success, message] = download_from_gdrive("1a2b3c4d5e6f7g8h9i0j", "gdrive_file.zip");
     */
    std::tuple<bool, std::string> download_from_gdrive(std::string gid, std::string outPath);
}