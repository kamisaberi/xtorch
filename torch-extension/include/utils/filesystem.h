#pragma once

#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace torch::ext::utils::filesystem {

    std::size_t countFiles(const fs::path& path , bool recursive = true);

}