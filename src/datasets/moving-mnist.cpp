#include "../../include/datasets/moving-mnist.h"

namespace xt::data::datasets {

    MovingMNIST::MovingMNIST(const std::string &root): MovingMNIST::MovingMNIST(root, DataMode::TRAIN, false) {
    }

    MovingMNIST::MovingMNIST(const std::string &root, DataMode mode): MovingMNIST::MovingMNIST(root, mode, false) {
    }

    MovingMNIST::MovingMNIST(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("MovingMNIST: MovingMNIST not implemented");
    }


    MovingMNIST::MovingMNIST(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("MovingMNIST: MovingMNIST not implemented");
    }


}
