#include "datasets/audio_processing/sound_event_detection/fsd50k.h"

namespace xt::data::datasets
{

    FSD50K::FSD50K(const std::string& root): FSD50K::FSD50K(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FSD50K::FSD50K(const std::string& root, xt::datasets::DataMode mode): FSD50K::FSD50K(
        root, mode, false, nullptr, nullptr)
    {
    }

    FSD50K::FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FSD50K::FSD50K(
            root, mode, download, nullptr, nullptr)
    {
    }

    FSD50K::FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FSD50K::FSD50K(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FSD50K::FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FSD50K::load_data()
    {

    }

    void FSD50K::check_resources()
    {

    }
}
