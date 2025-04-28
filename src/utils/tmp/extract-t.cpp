#include "../../include/utils/extract.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::utils {
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


    std::tuple<bool, string> extractGzip(const std::string &inFile, const std::string &outFile) {
        gzFile gz = gzopen(inFile.c_str(), "rb");
        if (!gz) {
            std::cerr << "Could not open inFile " << inFile << std::endl;
            return std::make_tuple(false, "");
        }


        string destPath = "";
        //    cout << "out file name: "<< fs::path(outFile).filename().string().size() << endl;
        if (outFile == "") {
            //destination file name and path didn't specify
            destPath = inFile.substr(0, inFile.size() - 3);
        } else if (fs::path(outFile).filename().string() == "") {
            //destination file name didn't specify but destination path specified
            string fln = fs::path(inFile).filename().string();
            string destFileName = fln.substr(0, fln.size() - 3);
            //        cout << destFileName << endl;
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

        // Extract all files from the tar archive
        cout << outPath << endl;
        //    if (tar_extract_all(tar, (char *) destPath.c_str()) == -1) { // Extract to current directory
        int res = tar_extract_all(tar, (char *) outPath.c_str());
        cout << "res:" << res;
        if (res == -1) {
            // Extract to current directory
            std::cerr << "Error extracting tar file: " << tarFile << std::endl;
            tar_close(tar);
            return true;
        }
        // Close the tar file
        tar_close(tar);
        std::cout << "Extraction completed successfully!" << std::endl;
        return true;
    }

    bool extractZip(const std::string &inFile, const std::string &outPath) {
        int err = 0;
        zip_t *zip_archive = zip_open(inFile.c_str(), ZIP_RDONLY, &err);

        if (!zip_archive) {
            //        char error_buffer[1024];
            zip_error_t zip_error;
            int error_code = 0;
            zip_error_set(&zip_error, error_code, 0);
            //        zip_error_to_str(error_buffer, sizeof(error_buffer), err, errno);
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
                // Create a new file to write the extracted content
                string path = (fs::path(destPath) / fs::path(filename)).string();
                FILE *output_file = fopen(path.c_str(), "wb");
                if (!output_file) {
                    std::cerr << "Failed to create file: " << path << std::endl;
                    zip_fclose(zip_file);
                    return false;
                }

                // Read the content from the ZIP file and write to the output file
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


    bool extractTgz(const std::string &filename, const std::string &output_dir) {
        struct archive *a = archive_read_new();
        struct archive *ext = archive_write_disk_new();
        struct archive_entry *entry;
        int flags = ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_ACL | ARCHIVE_EXTRACT_FFLAGS;
        int r;

        // Enable support for gzip and tar formats
        archive_read_support_filter_gzip(a);
        archive_read_support_format_tar(a);

        // Set up extraction options
        archive_write_disk_set_options(ext, flags);
        archive_write_disk_set_standard_lookup(ext);

        // Open the archive
        r = archive_read_open_filename(a, filename.c_str(), 10240); // 10KB block size
        if (r != ARCHIVE_OK) {
            std::cerr << "Failed to open archive: " << archive_error_string(a) << std::endl;
            return false;
        }

        // Create output directory if it doesn't exist
        fs::create_directories(output_dir);

        // Extraction loop
        while (true) {
            r = archive_read_next_header(a, &entry);
            if (r == ARCHIVE_EOF)
                break;
            if (r != ARCHIVE_OK) {
                std::cerr << "Error reading header: " << archive_error_string(a) << std::endl;
                return false;
            }

            // Construct full output path
            std::string output_path = output_dir + "/" + archive_entry_pathname(entry);
            archive_entry_set_pathname(entry, output_path.c_str());

            // Extract the entry
            r = archive_write_header(ext, entry);
            if (r != ARCHIVE_OK) {
                std::cerr << "Error writing header: " << archive_error_string(ext) << std::endl;
                return false;
            }

            // Write data if it's a file
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

        // Cleanup
        archive_read_close(a);
        archive_read_free(a);
        archive_write_close(ext);
        archive_write_free(ext);

        return true;
    }


    bool extract(const std::string &inFile, const std::string &outFile) {
        string ext = fs::path(inFile).filename().extension().string();
        //    cout << ext << endl;
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
}
