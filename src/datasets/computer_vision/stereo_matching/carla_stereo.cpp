#include "include/datasets/computer_vision/stereo_matching/carla_stereo.h"

namespace xt::datasets
{
    // ---------------------- CarlaStereo ---------------------- //

    CarlaStereo::CarlaStereo(const std::string& root): CarlaStereo::CarlaStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CarlaStereo::CarlaStereo(const std::string& root, xt::datasets::DataMode mode): CarlaStereo::CarlaStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    CarlaStereo::CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CarlaStereo::CarlaStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    CarlaStereo::CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CarlaStereo::CarlaStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CarlaStereo::CarlaStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CarlaStereo::load_data()
    {

    }

    void CarlaStereo::check_resources()
    {

    }
}
