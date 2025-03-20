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

namespace torch::ext::utils {
    void extractXZ(const std::string &inputFile, const std::string &outputFile);

    std::tuple<bool, string> extractGzip(const std::string &inFile, const std::string &outFile = "");

    bool extractTar(const std::string &tarFile, const std::string &outPath = "./");

    bool extractZip(const std::string &inFile, const std::string &outPath = "./");

    bool extractTgz(const std::string &inFile, const std::string &outPath = "./");

    bool extract(const std::string &inFile, const std::string &outFile = "");

}
