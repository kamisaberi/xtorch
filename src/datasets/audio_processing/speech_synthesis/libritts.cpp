#include "include/datasets/audio_processing/speech_synthesis/libritts.h"

namespace xt::data::datasets
{

    LIBRITTS::LIBRITTS(const std::string& root): LIBRITTS::LIBRITTS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LIBRITTS::LIBRITTS(const std::string& root, xt::datasets::DataMode mode): LIBRITTS::LIBRITTS(
        root, mode, false, nullptr, nullptr)
    {
    }

    LIBRITTS::LIBRITTS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LIBRITTS::LIBRITTS(
            root, mode, download, nullptr, nullptr)
    {
    }

    LIBRITTS::LIBRITTS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LIBRITTS::LIBRITTS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LIBRITTS::LIBRITTS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LIBRITTS::load_data()
    {

    }

    void LIBRITTS::check_resources()
    {

    }
}
