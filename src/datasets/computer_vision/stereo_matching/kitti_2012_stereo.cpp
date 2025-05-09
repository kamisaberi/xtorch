#include "../../../../include/datasets/computer_vision/stereo_matching/kitti_2012_stereo.h"

namespace xt::data::datasets {



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

}
