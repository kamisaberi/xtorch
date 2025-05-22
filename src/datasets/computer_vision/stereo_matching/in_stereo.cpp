#include "datasets/computer_vision/stereo_matching/in_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- InStereo2k ---------------------- //

    InStereo2k::InStereo2k(const std::string& root): InStereo2k::InStereo2k(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    InStereo2k::InStereo2k(const std::string& root, xt::datasets::DataMode mode): InStereo2k::InStereo2k(
        root, mode, false, nullptr, nullptr)
    {
    }

    InStereo2k::InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download) :
        InStereo2k::InStereo2k(
            root, mode, download, nullptr, nullptr)
    {
    }

    InStereo2k::InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : InStereo2k::InStereo2k(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    InStereo2k::InStereo2k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void InStereo2k::load_data()
    {

    }

    void InStereo2k::check_resources()
    {

    }
}
