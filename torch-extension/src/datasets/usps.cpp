#include "../../include/datasets/usps.h"

namespace xt::data::datasets {
    USPS::USPS(const std::string &root): USPS::USPS(root, DataMode::TRAIN, false) {
    }

    USPS::USPS(const std::string &root, DataMode mode): USPS::USPS(root, mode, false) {
    }

    USPS::USPS(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("USPS: USPS not implemented");
    }


    USPS::USPS(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("USPS: USPS not implemented");
    }


}
