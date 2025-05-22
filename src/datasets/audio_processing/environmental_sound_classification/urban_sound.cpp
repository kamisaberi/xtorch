#include "datasets/audio_processing/environmental_sound_classification/urban_sound.h"

namespace xt::data::datasets
{
    // ---------------------- UrbanSound ---------------------- //

    UrbanSound::UrbanSound(const std::string& root): UrbanSound::UrbanSound(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    UrbanSound::UrbanSound(const std::string& root, xt::datasets::DataMode mode): UrbanSound::UrbanSound(
        root, mode, false, nullptr, nullptr)
    {
    }

    UrbanSound::UrbanSound(const std::string& root, xt::datasets::DataMode mode, bool download) :
        UrbanSound::UrbanSound(
            root, mode, download, nullptr, nullptr)
    {
    }

    UrbanSound::UrbanSound(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : UrbanSound::UrbanSound(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    UrbanSound::UrbanSound(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void UrbanSound::load_data()
    {

    }

    void UrbanSound::check_resources()
    {

    }
}
