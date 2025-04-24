#include "../../../include/datasets/image_classification/omniglot.h"

namespace xt::data::datasets {

    Omniglot::Omniglot(const std::string &root): Omniglot(root, DataMode::TRAIN, false) {
    }

    Omniglot::Omniglot(const std::string &root, DataMode mode): Omniglot::Omniglot(root, mode, false) {
    }

    Omniglot::Omniglot(const std::string& root, DataMode mode, bool download) : BaseDataset(root, mode, download)
    {
        throw std::runtime_error("Omniglot: Omniglot not implemented");
    }


    Omniglot::Omniglot(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Omniglot: Omniglot not implemented");
    }


}
