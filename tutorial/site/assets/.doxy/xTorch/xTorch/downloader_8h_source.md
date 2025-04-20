

# File downloader.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**utils**](dir_821002d4f10779a80d4fb17bc32f21f1.md) **>** [**downloader.h**](downloader_8h.md)

[Go to the documentation of this file](downloader_8h.md)


```C++
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
    std::tuple<bool, std::string> download(std::string &url, std::string outPath);

    std::string rebuild_google_drive_link(std::string gid);

    std::tuple<bool, std::string> download_from_gdrive(std::string gid, std::string outPath);
}
```


