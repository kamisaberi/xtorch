

# File extract.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**utils**](dir_821002d4f10779a80d4fb17bc32f21f1.md) **>** [**extract.h**](extract_8h.md)

[Go to the documentation of this file](extract_8h.md)


```C++
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
    void extractXZ(const std::string &inputFile, const std::string &outputFile);

    std::tuple<bool, string> extractGzip(const std::string &inFile, const std::string &outFile = "");

    bool extractTar(const std::string &tarFile, const std::string &outPath = "./");

    bool extractZip(const std::string &inFile, const std::string &outPath = "./");

    bool extractTgz(const std::string &inFile, const std::string &outPath = "./");

    bool extract(const std::string &inFile, const std::string &outFile = "");

}
```


