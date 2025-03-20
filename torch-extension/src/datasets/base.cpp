#include "../../include/datasets/base.h"

namespace torch::ext::data::datasets {
    BaseDataset::BaseDataset(const std::string &root, DataMode mode, bool download) {
        this->root = root;
        this->download = download;
        this->mode = mode;
    }


}
