#include "../../include/datasets/sbu.h"

namespace xt::data::datasets {

    SBU::SBU(const std::string &root): SBU::SBU(root, DataMode::TRAIN, false) {
    }

    SBU::SBU(const std::string &root, DataMode mode): SBU::SBU(root, mode, false) {
    }

    SBU::SBU(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SBU: SBU not implemented");
    }


    SBU::SBU(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SBU: SBU not implemented");
    }


}
