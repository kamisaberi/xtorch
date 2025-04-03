#include "../../../include/datasets/base/base.h"

namespace xt::data::datasets {
    BaseDataset::BaseDataset(const std::string &root): BaseDataset::BaseDataset(root, DataMode::TRAIN, false) {
    }

    BaseDataset::BaseDataset(const std::string &root, DataMode mode): BaseDataset::BaseDataset(root, mode, false) {
    }

    BaseDataset::BaseDataset(const std::string &root, DataMode mode, bool download) : root(root), mode(mode),
        download(download) {
    }

    BaseDataset::BaseDataset(const std::string &root, DataMode mode, bool download,
                             TransformType transforms) : BaseDataset::BaseDataset(root, mode, download) {
        this->transforms = transforms;
        if (!transforms.empty()) {
            this->compose = xt::data::transforms::Compose(this->transforms);
        }
    }


    torch::data::Example<> BaseDataset::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> BaseDataset::size() const {
        return data.size();
    }
}
