#include "datasets/audio_processing/speech_recognition/libri_light_limited.h"

namespace xt::data::datasets
{

    LibriLightLimited::LibriLightLimited(const std::string& root): LibriLightLimited::LibriLightLimited(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LibriLightLimited::LibriLightLimited(const std::string& root, xt::datasets::DataMode mode): LibriLightLimited::LibriLightLimited(
        root, mode, false, nullptr, nullptr)
    {
    }

    LibriLightLimited::LibriLightLimited(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LibriLightLimited::LibriLightLimited(
            root, mode, download, nullptr, nullptr)
    {
    }

    LibriLightLimited::LibriLightLimited(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LibriLightLimited::LibriLightLimited(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LibriLightLimited::LibriLightLimited(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LibriLightLimited::load_data()
    {

    }

    void LibriLightLimited::check_resources()
    {

    }
}
