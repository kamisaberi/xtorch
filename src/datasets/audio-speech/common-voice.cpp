//
// Created by pc on 4/14/2025.
//

#include "../../../include/datasets/audio-speech/common-voice.h"

namespace xt::data::datasets {

    CommonVoice::CommonVoice(const std::string &root): CMUArctic::CMUArctic(root, DataMode::TRAIN, false) {
    }

    CommonVoice::CommonVoice(const std::string &root, DataMode mode): CMUArctic::CMUArctic(root, mode, false) {
    }

    CommonVoice::CommonVoice(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


    CMUArctic::CMUArctic(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("GTSRB: GTSRB not implemented");
    }


}
