#include "datasets/audio_processing/environmental_sound_classification/esc.h"

namespace xt::data::datasets
{
    // ---------------------- ESC ---------------------- //

    ESC::ESC(const std::string& root): ESC::ESC(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ESC::ESC(const std::string& root, xt::datasets::DataMode mode): ESC::ESC(
        root, mode, false, nullptr, nullptr)
    {
    }

    ESC::ESC(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ESC::ESC(
            root, mode, download, nullptr, nullptr)
    {
    }

    ESC::ESC(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ESC::ESC(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ESC::ESC(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ESC::load_data()
    {

    }

    void ESC::check_resources()
    {

    }
}
