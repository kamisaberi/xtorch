#include "../../../include/datasets/audio-speech/cmu-arctic.h"

namespace xt::data::datasets {

    CMUArctic::CMUArctic(const std::string &root): CMUArctic::CMUArctic(root, DataMode::TRAIN, false) {
    }

    CMUArctic::CMUArctic(const std::string &root, DataMode mode): CMUArctic::CMUArctic(root, mode, false) {
    }

    CMUArctic::CMUArctic(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


    CMUArctic::CMUArctic(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


}
