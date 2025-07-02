/**
 * @file extract.h
 * @brief Archive extraction utilities for various formats
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements extraction utilities for multiple archive formats:
 * - XZ/LZMA compressed files
 * - GZIP compressed files
 * - TAR archives
 * - ZIP archives
 * - TGZ (Gzipped TAR) archives
 * - Automatic format detection
 */

#include "../../include/utils/extract.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::utils {

    /**
     * @brief Extracts XZ/LZMA compressed files
     * @param inputFile Path to the input XZ file
     * @param outputFile Path for the decompressed output file
     * @throws std::runtime_error If extraction fails
     *
     * Uses LZMA SDK to decompress XZ files with:
     * - 8KB input/output buffers
     * - Error checking at each stage
     * - Proper resource cleanup
     */
    void extractXZ(const std::string &inputFile, const std::string &outputFile) {
        // Open the input file
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open input file: " + inputFile);
        }

        // Open the output file
        std::ofstream outFile(outputFile, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open output file: " + outputFile);
        }

        // Initialize LZMA stream
        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
        if (ret != LZMA_OK) {
            throw std::runtime_error("Failed to initialize LZMA decoder");
        }

        // Buffers for reading and writing
        std::vector<uint8_t> inBuffer(8192); // Input buffer
        std::vector<uint8_t> outBuffer(8192); // Output buffer

        // Decompress the file
        strm.next_in = nullptr;
        strm.avail_in = 0;
        strm.next_out = outBuffer.data();
        strm.avail_out = outBuffer.size();

        while (true) {
            // Read more data if the input buffer is empty
            if (strm.avail_in == 0 && !inFile.eof()) {
                inFile.read(reinterpret_cast<char *>(inBuffer.data()), inBuffer.size());
                strm.next_in = inBuffer.data();
                strm.avail_in = inFile.gcount();
            }

            // Decompress the data
            ret = lzma_code(&strm, inFile.eof() ? LZMA_FINISH : LZMA_RUN);

            // Write decompressed data to the output file
            if (strm.avail_out == 0 || ret == LZMA_STREAM_END) {
                size_t writeSize = outBuffer.size() - strm.avail_out;
                outFile.write(reinterpret_cast<char *>(outBuffer.data()), writeSize);
                strm.next_out = outBuffer.data();
                strm.avail_out = outBuffer.size();
            }

            // Check for errors or completion
            if (ret == LZMA_STREAM_END) {
                break; // Decompression completed
            } else if (ret != LZMA_OK) {
                lzma_end(&strm);
                throw std::runtime_error("LZMA decompression failed");
            }
        }

        // Clean up
        lzma_end(&strm);
        inFile.close();
        outFile.close();
    }

    /**
     * @brief Extracts GZIP compressed files
     * @param inFile Path to the input GZIP file
     * @param outFile Output path (empty for auto-determined path)
     * @return tuple<bool, string> (success status, output file path)
     *
     * Handles GZIP extraction with:
     * - Automatic output path determination
     * - 4KB read buffer
     * - Error logging
     */
    std::tuple<bool, string> extractGzip(const std::string &inFile, const std::string &outFile) {
        gzFile gz = gzopen(inFile.c_str(), "rb");
        if (!gz) {
            std::cerr << "Could not open inFile " << inFile << std::endl;
            return std::make_tuple(false, "");
        }

        string destPath = "";
        if (outFile == "") {
            //destination file name and path didn't specify
            destPath = inFile.substr(0, inFile.size() - 3);
        } else if (fs::path(outFile).filename().string() == "") {
            //destination file name didn't specify but destination path specified
            string fln = fs::path(inFile).filename().string();
            string destFileName = fln.substr(0, fln.size() - 3);
            destPath = (fs::path(outFile) / fs::path(destFileName)).string();
        } else if (fs::path(outFile).filename().string() == outFile) {
            //destination file name specified but path didn't specify
            destPath = "./" + outFile;
        }

        std::ofstream out(destPath, std::ios::binary);
        if (!out) {
            std::cerr << "Could not open detFile" << destPath << std::endl;
            gzclose(gz);
            return std::make_tuple(false, "");
        }

        char buffer[4096];
        int bytesRead;
        while ((bytesRead = gzread(gz, buffer, sizeof(buffer))) > 0) {
            out.write(buffer, bytesRead);
        }

        gzclose(gz);
        out.close();
        return std::make_tuple(true, destPath);
    }

    /**
     * @brief Extracts TAR archives
     * @param tarFile Path to the TAR archive
     * @param outPath Output directory (empty for current directory)
     * @return bool True if extraction succeeded
     *
     * Uses libtar to extract archives with:
     * - GNU TAR format support
     * - Basic error reporting
     */
    bool extractTar(const std::string &tarFile, const std::string &outPath) {
        TAR *tar;
        // Open the tar file
        if (tar_open(&tar, tarFile.c_str(), NULL, O_RDONLY, 0, TAR_GNU) == -1) {
            std::cerr << "Could not open " << tarFile << std::endl;
            return false;
        }

        string destPath = "./";
        if (outPath == "") {
            destPath = fs::path(tarFile).parent_path().string();
        }

        int res = tar_extract_all(tar, (char *) outPath.c_str());
        cout << "res:" << res;
        if (res == -1) {
            std::cerr << "Error extracting tar file: " << tarFile << std::endl;
            tar_close(tar);
            return true;
        }

        tar_close(tar);
        std::cout << "Extraction completed successfully!" << std::endl;
        return true;
    }

    /**
     * @brief Extracts ZIP archives
     * @param inFile Path to the ZIP archive
     * @param outPath Output directory (empty for archive directory)
     * @return bool True if extraction succeeded
     *
     * Uses libzip to extract archives with:
     * - Directory creation support
     * - 8KB read buffer
     * - Comprehensive error reporting
     */
    bool extractZip(const std::string &inFile, const std::string &outPath) {
        int err = 0;
        zip_t *zip_archive = zip_open(inFile.c_str(), ZIP_RDONLY, &err);

        if (!zip_archive) {
            zip_error_t zip_error;
            int error_code = 0;
            zip_error_set(&zip_error, error_code, 0);
            const char *error_buffer = zip_error_strerror(&zip_error);

            std::cerr << "Failed to open ZIP file: " << error_buffer << std::endl;
            return false;
        }

        zip_int64_t num_entries = zip_get_num_entries(zip_archive, 0);
        for (zip_int64_t i = 0; i < num_entries; i++) {
            const char *filename = zip_get_name(zip_archive, i, 0);
            if (!filename) {
                std::cerr << "Failed to get name for entry " << i << std::endl;
                continue;
            }

            string fname(filename);
            string destPath = "./";
            if (outPath == "") {
                destPath = fs::path(inFile).parent_path().string();
            }

            if (fname[fname.size() - 1] == '/') {
                fname = fname.substr(0, fname.length() - 1);
                std::error_code ec;
                string path = (fs::path(destPath) / fs::path(fname)).string();
                if (std::filesystem::create_directory(path, ec)) {
                    std::cout << "Directory created successfully: " << fname << std::endl;
                } else {
                    std::cout << "Failed to create directory: " << ec.message() << std::endl;
                }
            } else {
                std::cout << "Extracting: " << filename << std::endl;

                zip_file_t *zip_file = zip_fopen(zip_archive, filename, 0);
                if (!zip_file) {
                    std::cerr << "Failed to open file in ZIP: " << filename << std::endl;
                    return false;
                }

                string path = (fs::path(destPath) / fs::path(filename)).string();
                FILE *output_file = fopen(path.c_str(), "wb");
                if (!output_file) {
                    std::cerr << "Failed to create file: " << path << std::endl;
                    zip_fclose(zip_file);
                    return false;
                }

                char buffer[8192];
                zip_int64_t bytes_read;
                while ((bytes_read = zip_fread(zip_file, buffer, sizeof(buffer))) > 0) {
                    fwrite(buffer, 1, bytes_read, output_file);
                }

                fclose(output_file);
                zip_fclose(zip_file);
            }
        }
        zip_close(zip_archive);
        return true;
    }

    /**
     * @brief Extracts TGZ (Gzipped TAR) archives
     * @param filename Path to the TGZ archive
     * @param output_dir Output directory
     * @return bool True if extraction succeeded
     *
     * Uses libarchive to extract with:
     * - Automatic directory creation
     * - 10KB block size
     * - Preservation of file attributes
     * - Comprehensive error reporting
     */
    bool extractTgz(const std::string &filename, const std::string &output_dir) {
        struct archive *a = archive_read_new();
        struct archive *ext = archive_write_disk_new();
        struct archive_entry *entry;
        int flags = ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_ACL | ARCHIVE_EXTRACT_FFLAGS;
        int r;

        archive_read_support_filter_gzip(a);
        archive_read_support_format_tar(a);

        archive_write_disk_set_options(ext, flags);
        archive_write_disk_set_standard_lookup(ext);

        r = archive_read_open_filename(a, filename.c_str(), 10240);
        if (r != ARCHIVE_OK) {
            std::cerr << "Failed to open archive: " << archive_error_string(a) << std::endl;
            return false;
        }

        fs::create_directories(output_dir);

        while (true) {
            r = archive_read_next_header(a, &entry);
            if (r == ARCHIVE_EOF)
                break;
            if (r != ARCHIVE_OK) {
                std::cerr << "Error reading header: " << archive_error_string(a) << std::endl;
                return false;
            }

            std::string output_path = output_dir + "/" + archive_entry_pathname(entry);
            archive_entry_set_pathname(entry, output_path.c_str());

            r = archive_write_header(ext, entry);
            if (r != ARCHIVE_OK) {
                std::cerr << "Error writing header: " << archive_error_string(ext) << std::endl;
                return false;
            }

            if (archive_entry_size(entry) > 0) {
                const void *buff;
                size_t size;
                int64_t offset;

                while (true) {
                    r = archive_read_data_block(a, &buff, &size, &offset);
                    if (r == ARCHIVE_EOF)
                        break;
                    if (r != ARCHIVE_OK) {
                        std::cerr << "Error reading data: " << archive_error_string(a) << std::endl;
                        return false;
                    }
                    r = archive_write_data_block(ext, buff, size, offset);
                    if (r != ARCHIVE_OK) {
                        std::cerr << "Error writing data: " << archive_error_string(ext) << std::endl;
                        return false;
                    }
                }
            }

            r = archive_write_finish_entry(ext);
            if (r != ARCHIVE_OK) {
                std::cerr << "Error finishing entry: " << archive_error_string(ext) << std::endl;
                return false;
            }
        }

        archive_read_close(a);
        archive_read_free(a);
        archive_write_close(ext);
        archive_write_free(ext);

        return true;
    }

    /**
     * @brief Automatically detects and extracts archive formats
     * @param inFile Input archive file path
     * @param outFile Output path/directory
     * @return bool True if extraction succeeded
     *
     * Supports:
     * - .gz (GZIP)
     * - .zip (ZIP)
     * - .tgz (Gzipped TAR)
     * - .tar (TAR)
     *
     * Automatically chains GZIP and TAR extraction when needed.
     */
    bool extract(const std::string &inFile, const std::string &outFile) {
        string ext = fs::path(inFile).filename().extension().string();
        if (ext == ".gz") {
            cout << outFile << endl;
            auto [result, path] = extractGzip(inFile, outFile);
            if (result) {
                cout << outFile << endl;
                extractTar(path, outFile);
            }
        } else if (ext == ".zip") {
            extractZip(inFile, outFile);
        } else if (ext == ".tgz") {
            extractTgz(inFile, outFile);
        } else if (ext == ".tar") {
            extractTar(inFile, outFile);
        }
        return false;
    }
} // namespace xt::utils

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * Dependencies:
 * - liblzma: For XZ/LZMA decompression
 * - zlib: For GZIP decompression
 * - libtar: For TAR archive handling
 * - libzip: For ZIP archive handling
 * - libarchive: For TGZ archive handling
 *
 * License: MIT
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 */