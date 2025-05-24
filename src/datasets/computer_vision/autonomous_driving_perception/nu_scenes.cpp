#include "include/datasets/computer_vision/autonomous_driving_perception/nu_scenes.h"

namespace xt::datasets
{
    // ---------------------- NuScenes ---------------------- //

    NuScenes::NuScenes(const std::string& root): NuScenes::NuScenes(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    NuScenes::NuScenes(const std::string& root, xt::datasets::DataMode mode): NuScenes::NuScenes(
        root, mode, false, nullptr, nullptr)
    {
    }

    NuScenes::NuScenes(const std::string& root, xt::datasets::DataMode mode, bool download) :
        NuScenes::NuScenes(
            root, mode, download, nullptr, nullptr)
    {
    }

    NuScenes::NuScenes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : NuScenes::NuScenes(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    NuScenes::NuScenes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void NuScenes::load_data()
    {

    }

    void NuScenes::check_resources()
    {

    }
}
