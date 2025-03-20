#include "../../include/datasets/base.h"

namespace torch::ext::data::datasets {
    BaseDataset::BaseDataset(const std::string &root, DataMode mode, bool download) {
        this->root = root;
        this->download = download;
        this->mode = mode;
    }

    BaseDataset::BaseDataset(const fs::path &root, DatasetArguments args) {
        auto [mode , download , transforms] = args;
        this->root = root;
        this->download = download;
        this->mode = mode;
        this->transforms = transforms;
    }


}
