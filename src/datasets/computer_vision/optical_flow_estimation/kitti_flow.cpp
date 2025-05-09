#include "../../../include/datasets/specific/kitti.h"

namespace xt::data::datasets {


    // ---------------------- KittiFlow ---------------------- //
    KittiFlow::KittiFlow(const std::string &root): KittiFlow::KittiFlow(root, DataMode::TRAIN, false) {
    }

    KittiFlow::KittiFlow(const std::string &root, DataMode mode): KittiFlow::KittiFlow(root, mode, false) {
    }

    KittiFlow::KittiFlow(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("KittiFlow: KittiFlow not implemented");
    }


    KittiFlow::KittiFlow(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("KittiFlow: KittiFlow not implemented");
    }


}
