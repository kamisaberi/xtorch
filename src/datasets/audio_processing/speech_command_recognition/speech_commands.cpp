#include "datasets/audio_processing/speech_command_recognition/speech_commands.h"

namespace xt::data::datasets
{
    // ---------------------- SpeechCommands ---------------------- //

    SpeechCommands::SpeechCommands(const std::string& root): SpeechCommands::SpeechCommands(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SpeechCommands::SpeechCommands(const std::string& root, xt::datasets::DataMode mode): SpeechCommands::SpeechCommands(
        root, mode, false, nullptr, nullptr)
    {
    }

    SpeechCommands::SpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SpeechCommands::SpeechCommands(
            root, mode, download, nullptr, nullptr)
    {
    }

    SpeechCommands::SpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SpeechCommands::SpeechCommands(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SpeechCommands::SpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SpeechCommands::load_data()
    {

    }

    void SpeechCommands::check_resources()
    {

    }
}
