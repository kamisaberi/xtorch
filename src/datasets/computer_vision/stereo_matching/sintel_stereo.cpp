#include "datasets/computer_vision/stereo_matching/sintel_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- SintelStereo ---------------------- //

    SintelStereo::SintelStereo(const std::string& root): SintelStereo::SintelStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SintelStereo::SintelStereo(const std::string& root, xt::datasets::DataMode mode): SintelStereo::SintelStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    SintelStereo::SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SintelStereo::SintelStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    SintelStereo::SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SintelStereo::SintelStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SintelStereo::SintelStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SintelStereo::load_data()
    {

    }

    void SintelStereo::check_resources()
    {

    }
}
