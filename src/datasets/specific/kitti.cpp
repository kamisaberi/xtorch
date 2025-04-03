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




    // ---------------------- Kitti2012Stereo ---------------------- //
    Kitti2012Stereo::Kitti2012Stereo(const std::string &root): Kitti2012Stereo::Kitti2012Stereo(root, DataMode::TRAIN, false) {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string &root, DataMode mode): Kitti2012Stereo::Kitti2012Stereo(root, mode, false) {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Kitti2012Stereo: Kitti2012Stereo not implemented");
    }


    Kitti2012Stereo::Kitti2012Stereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Kitti2012Stereo: Kitti2012Stereo not implemented");
    }




    // ---------------------- Kitti2015Stereo ---------------------- //
    Kitti2015Stereo::Kitti2015Stereo(const std::string &root): Kitti2015Stereo::Kitti2015Stereo(root, DataMode::TRAIN, false) {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string &root, DataMode mode): Kitti2015Stereo::Kitti2015Stereo(root, mode, false) {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Kitti2015Stereo: Kitti2015Stereo not implemented");
    }


    Kitti2015Stereo::Kitti2015Stereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Kitti2015Stereo: Kitti2015Stereo not implemented");
    }


}
