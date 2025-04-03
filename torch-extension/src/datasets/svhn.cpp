#include "../../include/datasets/svhn.h"

namespace xt::data::datasets {

    SVHN::SVHN(const std::string &root): SVHN::SVHN(root, DataMode::TRAIN, false) {
    }

    SVHN::SVHN(const std::string &root, DataMode mode): SVHN::SVHN(root, mode, false) {
    }

    SVHN::SVHN(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SVHN: SVHN not implemented");
    }


    SVHN::SVHN(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SVHN: SVHN not implemented");
    }


}
