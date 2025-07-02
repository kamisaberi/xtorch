#include "../../include/utils/filesystem.h"

namespace xt::utils::fs {

    std::size_t countFiles(const std::filesystem::path& path , bool recursive){

        if (recursive) {
            return std::count_if(
                std::filesystem::recursive_directory_iterator(path),
                std::filesystem::recursive_directory_iterator{},
                [](const std::filesystem::directory_entry& entry) {
                    return entry.is_regular_file();
                }
            );


        }else {
            return std::count_if(
                std::filesystem::directory_iterator(path),
                std::filesystem::directory_iterator{},
                [](const std::filesystem::directory_entry& entry) {
                    return entry.is_regular_file();
                }
            );
        }
    }
}

