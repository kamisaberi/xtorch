#include "../../include/datasets/middlebury.h"

namespace xt::data::datasets {

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string &root): Middlebury2014Stereo::Middlebury2014Stereo(root, DataMode::TRAIN, false) {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string &root, DataMode mode): Middlebury2014Stereo::Middlebury2014Stereo(root, mode, false) {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Middlebury2014Stereo: Middlebury2014Stereo not implemented");
    }


    Middlebury2014Stereo::Middlebury2014Stereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Middlebury2014Stereo: Middlebury2014Stereo not implemented");
    }


}
