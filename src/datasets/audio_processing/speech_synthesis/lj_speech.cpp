#include "datasets/audio_processing/speech_synthesis/lj_speech.h"


namespace xt::data::datasets
{
    // ---------------------- LjSpeech ---------------------- //

    LjSpeech::LjSpeech(const std::string& root): LjSpeech::LjSpeech(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LjSpeech::LjSpeech(const std::string& root, xt::datasets::DataMode mode): LjSpeech::LjSpeech(
        root, mode, false, nullptr, nullptr)
    {
    }

    LjSpeech::LjSpeech(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LjSpeech::LjSpeech(
            root, mode, download, nullptr, nullptr)
    {
    }

    LjSpeech::LjSpeech(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LjSpeech::LjSpeech(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LjSpeech::LjSpeech(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LjSpeech::load_data()
    {

    }

    void LjSpeech::check_resources()
    {

    }
}
