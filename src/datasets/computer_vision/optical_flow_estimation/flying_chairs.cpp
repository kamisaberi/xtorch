#include "../../../../include/datasets/computer_vision/optical_flow_estimation/flying_chairs.h"

namespace xt::data::datasets {

    // ---------------------- FlyingChairs ---------------------- //

    FlyingChairs::FlyingChairs(const std::string &root): FlyingChairs::FlyingChairs(root, DataMode::TRAIN, false) {
    }

    FlyingChairs::FlyingChairs(const std::string &root, DataMode mode): FlyingChairs::FlyingChairs(root, mode, false) {
    }

    FlyingChairs::FlyingChairs(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("FlyingChairs: FlyingChairs not implemented");
    }


    FlyingChairs::FlyingChairs(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("FlyingChairs: FlyingChairs not implemented");
    }

}
