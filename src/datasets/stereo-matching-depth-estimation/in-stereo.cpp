#include "../../../include/datasets/stereo-matching-depth-estimation/in-stereo.h"

namespace xt::data::datasets {

    InStereo2k::InStereo2k(const std::string &root): InStereo2k::InStereo2k(root, DataMode::TRAIN, false) {
    }

    InStereo2k::InStereo2k(const std::string &root, DataMode mode): InStereo2k::InStereo2k(root, mode, false) {
    }

    InStereo2k::InStereo2k(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("InStereo2k: InStereo2k not implemented");
    }


    InStereo2k::InStereo2k(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("InStereo2k: InStereo2k not implemented");
    }


}
