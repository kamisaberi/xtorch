#pragma once

#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace xt::utils::fs {

    std::size_t countFiles(const std::filesystem::path& path , bool recursive = true);

}