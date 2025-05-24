#include "include/datasets/audio_processing/speaker_identification_and_verification/vox_celeb.h"

namespace xt::data::datasets
{
    // ---------------------- VoxCeleb ---------------------- //

    VoxCeleb::VoxCeleb(const std::string& root): VoxCeleb::VoxCeleb(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VoxCeleb::VoxCeleb(const std::string& root, xt::datasets::DataMode mode): VoxCeleb::VoxCeleb(
        root, mode, false, nullptr, nullptr)
    {
    }

    VoxCeleb::VoxCeleb(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VoxCeleb::VoxCeleb(
            root, mode, download, nullptr, nullptr)
    {
    }

    VoxCeleb::VoxCeleb(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VoxCeleb::VoxCeleb(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VoxCeleb::VoxCeleb(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VoxCeleb::load_data()
    {

    }

    void VoxCeleb::check_resources()
    {

    }
}
