#include "include/datasets/audio_processing/speech_recognition/tedlium.h"


namespace xt::datasets
{
    // ---------------------- Tedlium ---------------------- //

    Tedlium::Tedlium(const std::string& root): Tedlium::Tedlium(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Tedlium::Tedlium(const std::string& root, xt::datasets::DataMode mode): Tedlium::Tedlium(
        root, mode, false, nullptr, nullptr)
    {
    }

    Tedlium::Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Tedlium::Tedlium(
            root, mode, download, nullptr, nullptr)
    {
    }

    Tedlium::Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Tedlium::Tedlium(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Tedlium::Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Tedlium::load_data()
    {

    }

    void Tedlium::check_resources()
    {

    }
}
