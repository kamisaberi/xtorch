#include "datasets/audio_processing/speech_command_recognition/fluent_speech_commands.h"

namespace xt::data::datasets
{
    
    FluentSpeechCommands::FluentSpeechCommands(const std::string& root): FluentSpeechCommands::FluentSpeechCommands(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FluentSpeechCommands::FluentSpeechCommands(const std::string& root, xt::datasets::DataMode mode): FluentSpeechCommands::FluentSpeechCommands(
        root, mode, false, nullptr, nullptr)
    {
    }

    FluentSpeechCommands::FluentSpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FluentSpeechCommands::FluentSpeechCommands(
            root, mode, download, nullptr, nullptr)
    {
    }

    FluentSpeechCommands::FluentSpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FluentSpeechCommands::FluentSpeechCommands(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FluentSpeechCommands::FluentSpeechCommands(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FluentSpeechCommands::load_data()
    {

    }

    void FluentSpeechCommands::check_resources()
    {

    }
}
