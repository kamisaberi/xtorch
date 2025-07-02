#include "include/datasets/audio_processing/speech_recognition/common_voice.h"


namespace xt::datasets
{
    // ---------------------- CommonVoice ---------------------- //

    CommonVoice::CommonVoice(const std::string& root): CommonVoice::CommonVoice(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CommonVoice::CommonVoice(const std::string& root, xt::datasets::DataMode mode): CommonVoice::CommonVoice(
        root, mode, false, nullptr, nullptr)
    {
    }

    CommonVoice::CommonVoice(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CommonVoice::CommonVoice(
            root, mode, download, nullptr, nullptr)
    {
    }

    CommonVoice::CommonVoice(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CommonVoice::CommonVoice(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CommonVoice::CommonVoice(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CommonVoice::load_data()
    {

    }

    void CommonVoice::check_resources()
    {

    }
}
