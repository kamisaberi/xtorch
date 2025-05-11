#include "../../../include/datasets/sequence_tagging/udpos.h"

namespace xt::data::datasets {

    UDPOS::UDPOS(const std::string &root): UDPOS::UDPOS(root, DataMode::TRAIN, false) {
    }

    UDPOS::UDPOS(const std::string &root, DataMode mode): UDPOS::UDPOS(root, mode, false) {
    }

    UDPOS::UDPOS(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("UDPOS: UDPOS not implemented");
    }


    UDPOS::UDPOS(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("UDPOS: UDPOS not implemented");
    }


}
