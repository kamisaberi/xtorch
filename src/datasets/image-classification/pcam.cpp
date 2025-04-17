#include "../../../include/datasets/image-classification/pcam.h"

namespace xt::data::datasets {

    PCAM::PCAM(const std::string &root): PCAM::PCAM(root, DataMode::TRAIN, false) {
    }

    PCAM::PCAM(const std::string &root, DataMode mode): PCAM::PCAM(root, mode, false) {
    }

    PCAM::PCAM(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("PCAM: PCAM not implemented");
    }


    PCAM::PCAM(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("PCAM: PCAM not implemented");
    }


}
