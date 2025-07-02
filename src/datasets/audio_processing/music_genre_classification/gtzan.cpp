#include "include/datasets/audio_processing/music_genre_classification/gtzan.h"

namespace xt::datasets
{
    // ---------------------- GTZAN ---------------------- //

    GTZAN::GTZAN(const std::string& root): GTZAN::GTZAN(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    GTZAN::GTZAN(const std::string& root, xt::datasets::DataMode mode): GTZAN::GTZAN(
        root, mode, false, nullptr, nullptr)
    {
    }

    GTZAN::GTZAN(const std::string& root, xt::datasets::DataMode mode, bool download) :
        GTZAN::GTZAN(
            root, mode, download, nullptr, nullptr)
    {
    }

    GTZAN::GTZAN(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : GTZAN::GTZAN(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    GTZAN::GTZAN(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void GTZAN::load_data()
    {

    }

    void GTZAN::check_resources()
    {

    }
}
