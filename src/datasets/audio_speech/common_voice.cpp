#include "../../../include/datasets/audio_speech/common_voice.h"

namespace xt::data::datasets {

    CommonVoice::CommonVoice(const std::string &root): CommonVoice::CommonVoice(root, DataMode::TRAIN, false) {
    }

    CommonVoice::CommonVoice(const std::string &root, DataMode mode): CommonVoice::CommonVoice(root, mode, false) {
    }

    CommonVoice::CommonVoice(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CommonVoice: CommonVoice not implemented");
    }


    CommonVoice::CommonVoice(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CommonVoice: CommonVoice not implemented");
    }


}
