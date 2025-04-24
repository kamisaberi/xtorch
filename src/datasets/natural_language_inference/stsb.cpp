#include "../../../include/datasets/natural_language_inference/stsb.h"

namespace xt::data::datasets {

    STSB::STSB(const std::string &root): STSB::STSB(root, DataMode::TRAIN, false) {
    }

    STSB::STSB(const std::string &root, DataMode mode): STSB::STSB(root, mode, false) {
    }

    STSB::STSB(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("STSB: STSB not implemented");
    }


    STSB::STSB(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("STSB: STSB not implemented");
    }


}
