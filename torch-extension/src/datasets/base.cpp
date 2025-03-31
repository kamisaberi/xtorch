#include "../../include/datasets/base.h"

namespace xt::data::datasets {
    BaseDataset::BaseDataset(const std::string &root, DataMode mode, bool download) {
        this->root = root;
        this->download = download;
        this->mode = mode;
    }

    torch::data::Example<> BaseDataset::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> BaseDataset::size() const {
        return data.size();
    }


}
