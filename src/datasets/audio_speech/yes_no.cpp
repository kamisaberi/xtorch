#include "../../../include/datasets/audio-speech/yes-no.h"

namespace xt::data::datasets {

    YesNo::YesNo(const std::string &root): YesNo::YesNo(root, DataMode::TRAIN, false) {
    }

    YesNo::YesNo(const std::string &root, DataMode mode): YesNo::YesNo(root, mode, false) {
    }

    YesNo::YesNo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("YesNo: YesNo not implemented");
    }


    YesNo::YesNo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("YesNo: YesNo not implemented");
    }


}
