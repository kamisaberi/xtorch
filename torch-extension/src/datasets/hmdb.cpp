#include "../../include/datasets/hmdb.h"

namespace xt::data::datasets {

    HMDB51::HMDB51(const std::string &root): HMDB51::HMDB51(root, DataMode::TRAIN, false) {
    }

    HMDB51::HMDB51(const std::string &root, DataMode mode): HMDB51::HMDB51(root, mode, false) {
    }

    HMDB51::HMDB51(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("HMDB51: HMDB51 not implemented");
    }


    HMDB51::HMDB51(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("HMDB51: HMDB51 not implemented");
    }

}
