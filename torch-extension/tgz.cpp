#include <archive.h>
#include <archive_entry.h>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

bool extract_tgz(const std::string& filename, const std::string& output_dir) {
    struct archive* a = archive_read_new();
    struct archive* ext = archive_write_disk_new();
    struct archive_entry* entry;
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
            const void* buff;
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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <archive.tgz> <output_directory>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string output_dir = argv[2];

    if (extract_tgz(filename, output_dir)) {
        std::cout << "Successfully extracted " << filename << " to " << output_dir << std::endl;
        return 0;
    } else {
        std::cerr << "Failed to extract " << filename << std::endl;
        return 1;
    }
}