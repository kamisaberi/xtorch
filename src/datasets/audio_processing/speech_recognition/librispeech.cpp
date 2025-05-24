#include "include/datasets/audio_processing/speech_recognition/librispeech.h"


namespace xt::datasets
{
    // ---------------------- LibriSpeech ---------------------- //

    LibriSpeech::LibriSpeech(const std::string& root): LibriSpeech::LibriSpeech(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LibriSpeech::LibriSpeech(const std::string& root, xt::datasets::DataMode mode): LibriSpeech::LibriSpeech(
        root, mode, false, nullptr, nullptr)
    {
    }

    LibriSpeech::LibriSpeech(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LibriSpeech::LibriSpeech(
            root, mode, download, nullptr, nullptr)
    {
    }

    LibriSpeech::LibriSpeech(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LibriSpeech::LibriSpeech(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LibriSpeech::LibriSpeech(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LibriSpeech::load_data()
    {

    }

    void LibriSpeech::check_resources()
    {

    }
}
