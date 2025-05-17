#include "../../include/datasets/dataset.h"

namespace xt::datasets {
    Dataset::Dataset(const std::string &root): Dataset::Dataset(root, DataMode::TRAIN, false) {
    }

    Dataset::Dataset(const std::string &root, DataMode mode): Dataset::Dataset(root, mode, false) {
    }

    Dataset::Dataset(const std::string &root, DataMode mode, bool download) : root(root), mode(mode),
        download(download) {
    }

    Dataset::Dataset(const std::string &root, DataMode mode, bool download,
                             TransformType transforms) : Dataset::Dataset(root, mode, download) {
        this->transforms = transforms;
        if (!transforms.empty()) {
            this->compose = xt::transforms::Compose(this->transforms);
        }
    }



    // BaseDataset::~BaseDataset() {}



    torch::data::Example<> Dataset::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> Dataset::size() const {
        return data.size();
    }
}
