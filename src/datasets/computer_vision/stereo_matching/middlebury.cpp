#include "datasets/computer_vision/stereo_matching/middlebury.h"

namespace xt::data::datasets
{
    // ---------------------- Middlebury2014Stereo ---------------------- //

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string& root): Middlebury2014Stereo::Middlebury2014Stereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string& root, xt::datasets::DataMode mode): Middlebury2014Stereo::Middlebury2014Stereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Middlebury2014Stereo::Middlebury2014Stereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Middlebury2014Stereo::Middlebury2014Stereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Middlebury2014Stereo::Middlebury2014Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Middlebury2014Stereo::load_data()
    {

    }

    void Middlebury2014Stereo::check_resources()
    {

    }
}
