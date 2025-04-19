#include "../../../include/datasets/audio-speech/lj-speech.h"

namespace xt::data::datasets {

    LjSpeech::LjSpeech(const std::string &root): CMUArctic::CMUArctic(root, DataMode::TRAIN, false) {
    }

    LjSpeech::LjSpeech(const std::string &root, DataMode mode): CMUArctic::CMUArctic(root, mode, false) {
    }

    LjSpeech::LjSpeech(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("LjSpeech: LjSpeech not implemented");
    }


    LjSpeech::LjSpeech(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("LjSpeech: LjSpeech not implemented");
    }


}
