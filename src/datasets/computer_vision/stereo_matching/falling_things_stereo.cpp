#include "include/datasets/computer_vision/stereo_matching/falling_things_stereo.h"

namespace xt::datasets
{
    // ---------------------- FallingThingsStereo ---------------------- //

    FallingThingsStereo::FallingThingsStereo(const std::string& root): FallingThingsStereo::FallingThingsStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string& root, xt::datasets::DataMode mode): FallingThingsStereo::FallingThingsStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FallingThingsStereo::FallingThingsStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FallingThingsStereo::FallingThingsStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FallingThingsStereo::FallingThingsStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FallingThingsStereo::load_data()
    {

    }

    void FallingThingsStereo::check_resources()
    {

    }
}
