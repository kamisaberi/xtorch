#include "../../../include/datasets/specific/kitti.h"

namespace xt::data::datasets {

    // ---------------------- Kitti ---------------------- //
    Kitti::Kitti(const std::string &root): Kitti::Kitti(root, DataMode::TRAIN, false) {
    }

    Kitti::Kitti(const std::string &root, DataMode mode): Kitti::Kitti(root, mode, false) {
    }

    Kitti::Kitti(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Kitti: Kitti not implemented");
    }


    Kitti::Kitti(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Kitti: Kitti not implemented");
    }





}
