#include "datasets/audio_processing/music_tagging/magna_tag_a_tune.h"

namespace xt::data::datasets
{
    // ---------------------- MagnaTagATune ---------------------- //

    MagnaTagATune::MagnaTagATune(const std::string& root): MagnaTagATune::MagnaTagATune(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MagnaTagATune::MagnaTagATune(const std::string& root, xt::datasets::DataMode mode): MagnaTagATune::MagnaTagATune(
        root, mode, false, nullptr, nullptr)
    {
    }

    MagnaTagATune::MagnaTagATune(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MagnaTagATune::MagnaTagATune(
            root, mode, download, nullptr, nullptr)
    {
    }

    MagnaTagATune::MagnaTagATune(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MagnaTagATune::MagnaTagATune(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MagnaTagATune::MagnaTagATune(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MagnaTagATune::load_data()
    {

    }

    void MagnaTagATune::check_resources()
    {

    }
}
