#include "../../../include/datasets/audio-speech/gtzan.h"

namespace xt::data::datasets {

    GTZAN::GTZAN(const std::string &root): CMUArctic::CMUArctic(root, DataMode::TRAIN, false) {
    }

    GTZAN::GTZAN(const std::string &root, DataMode mode): CMUArctic::CMUArctic(root, mode, false) {
    }

    GTZAN::GTZAN(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("GTZAN: GTZAN not implemented");
    }


    GTZAN::GTZAN(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("GTZAN: GTZAN not implemented");
    }


}
