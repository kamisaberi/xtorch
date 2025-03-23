#include "../../include/utils/filesystem.h"

namespace torch::est::utils::filesystem {

    std::size_t countFiles(const fs::path& path , bool recursive){

      if (recursive) {

          return std::count_if(
      std::filesystem::recursive_directory_iterator(path),
      std::filesystem::recursive_directory_iterator{},
      [](const std::filesystem::directory_entry& entry) {
          return entry.is_regular_file();
      }
  );

      }


    }

}

