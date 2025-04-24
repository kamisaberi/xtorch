#include "../../../include/datasets/natural_language_inference/rte.h"

namespace xt::data::datasets {

    RTE::RTE(const std::string &root): RTE::RTE(root, DataMode::TRAIN, false) {
    }

    RTE::RTE(const std::string &root, DataMode mode): RTE::RTE(root, mode, false) {
    }

    RTE::RTE(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("RTE: RTE not implemented");
    }


    RTE::RTE(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("RTE: RTE not implemented");
    }


}
