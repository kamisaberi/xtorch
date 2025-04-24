#include "../../../include/datasets/image_classification/gtsrb.h"

namespace xt::data::datasets {

    GTSRB::GTSRB(const std::string &root): GTSRB::GTSRB(root, DataMode::TRAIN, false) {
    }

    GTSRB::GTSRB(const std::string &root, DataMode mode): GTSRB::GTSRB(root, mode, false) {
    }

    GTSRB::GTSRB(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


    GTSRB::GTSRB(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


}
