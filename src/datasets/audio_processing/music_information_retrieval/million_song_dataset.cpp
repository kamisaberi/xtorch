#include "datasets/audio_processing/music_information_retrieval/million_song_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- MillionSongDataset ---------------------- //

    MillionSongDataset::MillionSongDataset(const std::string& root): MillionSongDataset::MillionSongDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MillionSongDataset::MillionSongDataset(const std::string& root, xt::datasets::DataMode mode): MillionSongDataset::MillionSongDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    MillionSongDataset::MillionSongDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MillionSongDataset::MillionSongDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    MillionSongDataset::MillionSongDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MillionSongDataset::MillionSongDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MillionSongDataset::MillionSongDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MillionSongDataset::load_data()
    {

    }

    void MillionSongDataset::check_resources()
    {

    }
}
