#include "../../../include/datasets/audio_speech/librispeech.h"

namespace xt::data::datasets {

    LibriSpeech::LibriSpeech(const std::string &root): LibriSpeech::LibriSpeech(root, DataMode::TRAIN, false) {
    }

    LibriSpeech::LibriSpeech(const std::string &root, DataMode mode): LibriSpeech::LibriSpeech(root, mode, false) {
    }

    LibriSpeech::LibriSpeech(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("LibriSpeech: LibriSpeech not implemented");
    }


    LibriSpeech::LibriSpeech(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("LibriSpeech: GTSRB not implemented");
    }


}
