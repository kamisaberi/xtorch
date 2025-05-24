#include "include/datasets/audio_processing/audio_event_detection/audioset.h"

namespace xt::data::datasets
{
    // ---------------------- Caltech101 ---------------------- //

    AudioSet::AudioSet(const std::string& root): AudioSet::AudioSet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AudioSet::AudioSet(const std::string& root, xt::datasets::DataMode mode): AudioSet::AudioSet(
        root, mode, false, nullptr, nullptr)
    {
    }

    AudioSet::AudioSet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AudioSet::AudioSet(
            root, mode, download, nullptr, nullptr)
    {
    }

    AudioSet::AudioSet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AudioSet::AudioSet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AudioSet::AudioSet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AudioSet::load_data()
    {

    }

    void AudioSet::check_resources()
    {

    }
}
