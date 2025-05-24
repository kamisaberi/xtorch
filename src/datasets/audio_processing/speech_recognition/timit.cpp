#include "include/datasets/audio_processing/speech_recognition/timit.h"


namespace xt::datasets
{
    // ---------------------- TIMIT ---------------------- //

    TIMIT::TIMIT(const std::string& root): TIMIT::TIMIT(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    TIMIT::TIMIT(const std::string& root, xt::datasets::DataMode mode): TIMIT::TIMIT(
        root, mode, false, nullptr, nullptr)
    {
    }

    TIMIT::TIMIT(const std::string& root, xt::datasets::DataMode mode, bool download) :
        TIMIT::TIMIT(
            root, mode, download, nullptr, nullptr)
    {
    }

    TIMIT::TIMIT(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : TIMIT::TIMIT(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    TIMIT::TIMIT(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void TIMIT::load_data()
    {

    }

    void TIMIT::check_resources()
    {

    }
}
