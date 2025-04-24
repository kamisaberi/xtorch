#include "../../../include/datasets/audio_speech/timit.h"

namespace xt::data::datasets {

    TIMIT::TIMIT(const std::string &root): TIMIT::TIMIT(root, DataMode::TRAIN, false) {
    }

    TIMIT::TIMIT(const std::string &root, DataMode mode): TIMIT::TIMIT(root, mode, false) {
    }

    TIMIT::TIMIT(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("TIMIT: TIMIT not implemented");
    }


    TIMIT::TIMIT(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("TIMIT: TIMIT not implemented");
    }


}
