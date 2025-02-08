#include <iostream>
#include <libtar.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <filesystem>


void extractTar(const std::string& tarFile) {
    TAR *tar;
    // Open the tar file
    if (tar_open(&tar, tarFile.c_str(), NULL, O_RDONLY, 0, TAR_GNU) == -1) {
        std::cerr << "Could not open " << tarFile << std::endl;
        return;
    }

    // Extract all files from the tar archive
    if (tar_extract_all(tar, "./") == -1) { // Extract to current directory
        std::cerr << "Error extracting tar file: " << tarFile << std::endl;
        tar_close(tar);
        return;
    }

    // Close the tar file
    tar_close(tar);
    std::cout << "Extraction completed successfully!" << std::endl;
}


int main() {
    const char *tarFile = "test.tar";  // Path to the .tar file
    const char *extractDir = "./"; // Destination directory

    extractTar(tarFile);
//    extractTar(tarFile, extractDir);
    return 0;
}
