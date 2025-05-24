#include "include/datasets/audio_processing/speech_separation/libri_mix.h"

namespace xt::data::datasets
{
    
    LibriMix::LibriMix(const std::string& root): LibriMix::LibriMix(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LibriMix::LibriMix(const std::string& root, xt::datasets::DataMode mode): LibriMix::LibriMix(
        root, mode, false, nullptr, nullptr)
    {
    }

    LibriMix::LibriMix(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LibriMix::LibriMix(
            root, mode, download, nullptr, nullptr)
    {
    }

    LibriMix::LibriMix(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LibriMix::LibriMix(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LibriMix::LibriMix(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LibriMix::load_data()
    {

    }

    void LibriMix::check_resources()
    {

    }
}
