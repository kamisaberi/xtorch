#include "../../../include/datasets/specific/kinetics.h"

namespace xt::data::datasets {

    Kinetics::Kinetics(const std::string &root): Kinetics::Kinetics(root, DataMode::TRAIN, false) {
    }

    Kinetics::Kinetics(const std::string &root, DataMode mode): Kinetics::Kinetics(root, mode, false) {
    }

    Kinetics::Kinetics(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Kinetics: Kinetics not implemented");
    }


    Kinetics::Kinetics(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Kinetics: Kinetics not implemented");
    }


}
