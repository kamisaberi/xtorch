#include "../../../include/datasets/audio-speech/vctk.h"

namespace xt::data::datasets {

    VCTK::VCTK(const std::string &root): VCTK::VCTK(root, DataMode::TRAIN, false) {
    }

    VCTK::VCTK(const std::string &root, DataMode mode): VCTK::VCTK(root, mode, false) {
    }

    VCTK::VCTK(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("VCTK: VCTK not implemented");
    }


    VCTK::VCTK(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("VCTK: VCTK not implemented");
    }


}
