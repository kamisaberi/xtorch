#include "../../include/datasets/flying.h"

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






    // ---------------------- FlyingThings3D ---------------------- //


    FlyingThings3D::FlyingThings3D(const std::string &root): FlyingThings3D::FlyingThings3D(root, DataMode::TRAIN, false) {
    }

    FlyingThings3D::FlyingThings3D(const std::string &root, DataMode mode): FlyingThings3D::FlyingThings3D(root, mode, false) {
    }

    FlyingThings3D::FlyingThings3D(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("FlyingThings3D: FlyingThings3D not implemented");
    }


    FlyingThings3D::FlyingThings3D(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("FlyingThings3D: FlyingThings3D not implemented");
    }

}
