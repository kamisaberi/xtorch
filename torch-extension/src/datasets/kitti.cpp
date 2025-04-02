#include "../../include/datasets/kitti.h"

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





    // ---------------------- Caltech101 ---------------------- //
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




    // ---------------------- Caltech101 ---------------------- //
    Caltech101::Caltech101(const std::string &root): Caltech101::Caltech101(root, DataMode::TRAIN, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode): Caltech101::Caltech101(root, mode, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }




    // ---------------------- Caltech101 ---------------------- //
    Caltech101::Caltech101(const std::string &root): Caltech101::Caltech101(root, DataMode::TRAIN, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode): Caltech101::Caltech101(root, mode, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


}
