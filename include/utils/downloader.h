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
