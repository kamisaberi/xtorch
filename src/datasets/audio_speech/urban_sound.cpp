#include "../../../include/datasets/audio_speech/urban_sound.h"

namespace xt::data::datasets {

    UrbanSound::UrbanSound(const std::string &root): UrbanSound::UrbanSound(root, DataMode::TRAIN, false) {
    }

    UrbanSound::UrbanSound(const std::string &root, DataMode mode): UrbanSound::UrbanSound(root, mode, false) {
    }

    UrbanSound::UrbanSound(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("UrbanSound: UrbanSound not implemented");
    }


    UrbanSound::UrbanSound(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("UrbanSound: UrbanSound not implemented");
    }


}
