#include "include/datasets/audio_processing/intent_classification/snips.h"

namespace xt::data::datasets
{
    // ---------------------- Caltech101 ---------------------- //

    Snips::Snips(const std::string& root): Snips::Snips(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Snips::Snips(const std::string& root, xt::datasets::DataMode mode): Snips::Snips(
        root, mode, false, nullptr, nullptr)
    {
    }

    Snips::Snips(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Snips::Snips(
            root, mode, download, nullptr, nullptr)
    {
    }

    Snips::Snips(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Snips::Snips(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Snips::Snips(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Snips::load_data()
    {

    }

    void Snips::check_resources()
    {

    }
}
