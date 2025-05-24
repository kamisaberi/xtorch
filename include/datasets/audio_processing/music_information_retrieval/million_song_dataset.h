#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets {
    class MillionSongDataset : public xt::datasets::Dataset {
    public :
        explicit MillionSongDataset(const std::string &root);

        MillionSongDataset(const std::string &root, xt::datasets::DataMode mode);

        MillionSongDataset(const std::string &root, xt::datasets::DataMode mode, bool download);

        MillionSongDataset(const std::string &root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr <xt::Module> transformer);

        MillionSongDataset(const std::string &root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr <xt::Module> transformer,
                           std::unique_ptr <xt::Module> target_transformer);

    private:

        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
