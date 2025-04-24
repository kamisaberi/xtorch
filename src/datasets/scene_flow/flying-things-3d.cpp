#include "../../../include/datasets/scene_flow/flying_things_3d.h"

namespace xt::data::datasets {


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
