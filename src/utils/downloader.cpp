/**
 * @file downloader.cpp
 * @brief File downloader implementation using libcurl
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements file download functionality including:
 * - Regular HTTP/HTTPS downloads
 * - Google Drive file downloads
 * - Progress tracking (commented out)
 * - File size detection (commented out)
 */

#include "../../include/utils/downloader.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::utils {

    /**
     * @brief Callback function for processing HTTP headers
     * @param buffer Pointer to received header data
     * @param size Size of each data element
     * @param nitems Number of elements
     * @param userdata Pointer to store extracted content length
     * @return size_t Total size of processed data
     *
     * This callback processes HTTP headers to extract Content-Length information.
     * Currently not used in the main download function (commented out).
     */
    size_t header_callback(char *buffer, size_t size, size_t nitems, void *userdata) {
        std::string header(buffer, size * nitems);

        // Look for the Content-Length header
        if (header.find("Content-Length:") != std::string::npos) {
            size_t pos = header.find(":") + 1;
            std::string length = header.substr(pos);
            // Trim whitespace
            length.erase(0, length.find_first_not_of(" \t\n\r"));
            length.erase(length.find_last_not_of(" \t\n\r") + 1);

            // Convert to size_t and store in userdata
            *(static_cast<size_t *>(userdata)) = std::stoul(length);
        }

        return nitems * size;
    }

    /**
     * @brief Reconstructs Google Drive download URL from file ID
     * @param gid Google Drive file ID
     * @return std::string Constructed download URL
     *
     * Converts a Google Drive file ID into a direct download URL.
     */
    std::string rebuild_google_drive_link(std::string gid) {
        return "https://drive.google.com/uc?id=" + gid;
    }

    /**
     * @brief Downloads a file from a given URL
     * @param url The URL to download from
     * @param outPath Output directory path
     * @return std::tuple<bool, std::string>
     *         (success status, downloaded file path)
     *
     * Uses libcurl to download files with the following features:
     * - Supports HTTP/HTTPS protocols
     * - Preserves original filename
     * - Saves to specified output directory
     * - Basic error handling
     */
    std::tuple<bool, std::string> download(std::string &url, std::string outPath) {
        cout << url << endl;
        CURL *curl;
        FILE *fp;
        CURLcode res;
        size_t file_size = 0;

        curl = curl_easy_init();
        if (curl) {
            string outFile = (fs::path(outPath) / fs::path(url).filename()).string();
            fp = fopen(outFile.c_str(), "wb");
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

            // Uncomment to enable file size detection
            // curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
            // curl_easy_setopt(curl, CURLOPT_HEADERDATA, &file_size);
            // cout << "FILE_SIZE:" <<file_size << endl;

            // Uncomment to enable progress tracking
            // curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            // curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, ProgressCallback);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
            res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);
            fclose(fp);
            return std::make_tuple(true, outFile);
        }
        return std::make_tuple(false, "");
    }

    /**
     * @brief Downloads a file from Google Drive
     * @param gid Google Drive file ID
     * @param outPath Output directory path
     * @return std::tuple<bool, std::string>
     *         (success status, downloaded file path)
     *
     * Specialized version for Google Drive downloads that:
     * - Converts Google Drive ID to proper URL
     * - Uses the standard download function
     */
    std::tuple<bool, std::string> download_from_gdrive(std::string gid, std::string outPath) {
        string url = rebuild_google_drive_link(gid);
        auto [result, path] = download(url, outPath);
        return std::make_tuple(result, path);
    }
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
 * - libcurl: For HTTP/HTTPS download functionality
 * - C++17 filesystem: For path manipulation
 */