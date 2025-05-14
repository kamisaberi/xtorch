#include "datasets/computer_vision/stereo_matching/kitti_2015_stereo.h"

namespace xt::data::datasets {


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
