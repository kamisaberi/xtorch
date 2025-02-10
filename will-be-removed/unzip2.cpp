#include <iostream>
#include <zip.h>
#include <filesystem>
#include <zlib.h>
#include <fstream>
#include <tar.h> // You may need to find a suitable tar library or implement tar parsing
#include <archive.h>
#include <archive_entry.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <libtar.h> // Make sure you have libtar installed
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>


using namespace  std;

void decompressGzip(const std::string& inFile, const std::string& outFile) {
    gzFile gz = gzopen(inFile.c_str(), "rb");
    if (!gz) {
        std::cerr << "Could not open " << inFile << std::endl;
        return;
    }

    std::ofstream out(outFile, std::ios::binary);
    if (!out) {
        std::cerr << "Could not open " << outFile << std::endl;
        gzclose(gz);
        return;
    }

    char buffer[4096];
    int bytesRead;
    while ((bytesRead = gzread(gz, buffer, sizeof(buffer))) > 0) {
        out.write(buffer, bytesRead);
    }

    gzclose(gz);
    out.close();
}





void extract_zip(string zip_filename) {
    int err = 0;
    zip_t* zip_archive = zip_open(zip_filename.c_str(), ZIP_RDONLY, &err);


    if (!zip_archive) {
        char error_buffer[1024];
        zip_error_to_str(error_buffer, sizeof(error_buffer), err, errno);
        std::cerr << "Failed to open ZIP file: " << error_buffer << std::endl;
        return;
    }

    zip_int64_t num_entries = zip_get_num_entries(zip_archive, 0);
    for (zip_int64_t i = 0; i < num_entries; i++) {

        const char* filename = zip_get_name(zip_archive, i, 0);
        if (!filename) {
            std::cerr << "Failed to get name for entry " << i << std::endl;
            continue;
        }

        string fname(filename);
        if (fname[fname.size() - 1] == '/') {

            fname = fname.substr(0,fname.length()-1);
            std::error_code ec;
            if (std::filesystem::create_directory(fname, ec)) {
                std::cout << "Directory created successfully: " << fname << std::endl;
            } else {
                std::cout << "Failed to create directory: " << ec.message() << std::endl;
            }

        } else {

            std::cout << "Extracting: " << filename << std::endl;
            zip_file_t* zip_file = zip_fopen(zip_archive, filename, 0);
            if (!zip_file) {
                std::cerr << "Failed to open file in ZIP: " << filename << std::endl;
                continue;
            }

            // Create a new file to write the extracted content
            FILE* output_file = fopen(filename, "wb");
            if (!output_file) {
                std::cerr << "Failed to create file: " << filename << std::endl;
                zip_fclose(zip_file);
                continue;
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
}

//void extractTar(const std::string& tarFile) {
//    struct archive *archivePtr = archive_read_new();
//    struct archive_entry *entry;
//
//    // Support tar format and all filters
//    archive_read_support_format_tar(archivePtr);
//    archive_read_support_filter_all(archivePtr);
//
//    // Open the tar file
//    if (archive_read_open_filename(archivePtr, tarFile.c_str(), 10240) != ARCHIVE_OK) {
//        std::cerr << "Could not open " << tarFile << ": " << archive_error_string(archivePtr) << std::endl;
//        return;
//    }
//
//    // Read each entry in the tar file
//    while (archive_read_next_header(archivePtr, &entry) == ARCHIVE_OK) {
//        const char* currentFile = archive_entry_pathname(entry);
//        std::cout << "Extracting: " << currentFile << std::endl;
//
//        // Create directories if necessary
//        if (archive_entry_filetype(entry) == AE_IFDIR) {
//            mkdir(currentFile, 0755); // Create directory with 755 permissions
//            continue;
//        }
//
//        // Create a file to extract the contents
//        std::ofstream outFile(currentFile, std::ios::binary);
//        if (!outFile) {
//            std::cerr << "Could not create " << currentFile << std::endl;
//            continue;
//        }
//
//        // Read the contents and write to the file
//        const void* buffer;
//        size_t size;
//        la_int64_t offset;
//
//        while (true) {
//            int result = archive_read_data_block(archivePtr, &buffer, &size, &offset);
//            if (result == ARCHIVE_EOF) {
//                break;
//            }
//            if (result != ARCHIVE_OK) {
//                std::cerr << "Error reading data: " << archive_error_string(archivePtr) << std::endl;
//                break;
//            }
//            outFile.write(static_cast<const char*>(buffer), size);
//        }
//
//        outFile.close();
//        archive_read_data_skip(archivePtr); // Skip to the next entry
//    }
//
//    // Clean up
//    archive_read_free(archivePtr);
//}


//void extractTar(const std::string& tarFile) {
//    TAR *tar;
//
//    // Open the tar file
//    if (tar_open(&tar, tarFile.c_str(), NULL, 0, 0, TAR_GNU) == -1) {
//        std::cerr << "Could not open " << tarFile << std::endl;
//        return;
//    }
//
//    // Extract all files from the tar archive
//    if (tar_extract_all(tar, "./") == -1) { // Extract to current directory
//        std::cerr << "Could not extract " << tarFile << std::endl;
//    }
//
//    // Close the tar file
//    tar_close(tar);
//}


//void extractTar(const std::string& tarFile) {
//    TAR *tar;
//    // Open the tar file
//    if (tar_open(&tar, tarFile.c_str(), NULL, 0, 0, TAR_GNU) == -1) {
//        std::cerr << "Could not open " << tarFile << std::endl;
//        return;
//    }
//
//    // Extract all files from the tar archive
//    if (tar_extract_all(tar, "./") == -1) { // Extract to current directory
//        std::cerr << "Error extracting tar file: " << tarFile << std::endl;
//        tar_close(tar);
//        return;
//    }
//
//    // Close the tar file
//    tar_close(tar);
//    std::cout << "Extraction completed successfully!" << std::endl;
//}


void extractTar(const char *tarFilename, const char *destinationDir) {
    TAR *tar;

    // Open the tar file
    if (tar_open(&tar, tarFilename, NULL, O_RDONLY, 0, TAR_GNU) == -1) {
        std::cerr << "Error: Failed to open TAR file.\n";
        return;
    }

    // Extract the tar file to the specified directory
    if (tar_extract_all(tar, (char*)destinationDir) == -1) {
        std::cerr << "Error: Failed to extract TAR file.\n";
        tar_close(tar);
        return;
    }

    // Close the tar file
    tar_close(tar);
    std::cout << "Extraction complete!\n";
}


//void extractTar(const char *tarFilename, const char *destinationDir) {
//    TAR *tar;
//
//    // Open the tar file
//    if (tar_open(&tar, tarFilename, NULL, O_RDONLY, 0, TAR_GNU) == -1) {
//        std::cerr << "Error: Failed to open TAR file.\n";
//        return;
//    }
//
//    // Iterate over each file in the tar archive
//    while (th_read(tar) == 0) {
//        // Construct the full output path
//        std::string outFilePath = std::string(destinationDir) + "/" + th_get_pathname(tar);
//
//        // Extract the current file or directory
//        if (tar_extract_file(tar, outFilePath.c_str()) == -1) {
//            std::cerr << "Error: Failed to extract " << th_get_pathname(tar) << "\n";
//            tar_close(tar);
//            return;
//        }
//
//        // Move to the next file in the tar
//        if (tar_skip_regfile(tar) == -1) {
//            std::cerr << "Error: Failed to skip file.\n";
//            tar_close(tar);
//            return;
//        }
//    }
//
//    // Close the tar file
//    tar_close(tar);
//    std::cout << "Extraction complete!\n";
//}





int main(int argc, char* argv[]) {
//    if (argc < 2) {
//        std::cerr << "Usage: " << argv[0] << " <zip_file>" << std::endl;
//        return 1;
//    }
    string zipFileName("/home/kami/Documents/a.tar.gz");



//    extract_zip(argv[1]);
//    extract_zip(zipFileName);
//    decompressGzip(zipFileName , "./test.tar");
    extractTar("test.tar", (const char* )"./");
    return 0;
}
