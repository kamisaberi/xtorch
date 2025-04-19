#include "../../../include/datasets/audio-speech/speech-commands.h"

namespace xt::data::datasets {

    SpeechCommands::SpeechCommands(const std::string &root): SpeechCommands::SpeechCommands(root, DataMode::TRAIN, false) {
    }

    SpeechCommands::SpeechCommands(const std::string &root, DataMode mode): SpeechCommands::SpeechCommands(root, mode, false) {
    }

    SpeechCommands::SpeechCommands(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SpeechCommands: SpeechCommands not implemented");
    }


    SpeechCommands::SpeechCommands(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SpeechCommands: SpeechCommands not implemented");
    }


}
