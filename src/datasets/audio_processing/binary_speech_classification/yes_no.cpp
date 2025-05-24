#include "include/datasets/audio_processing/binary_speech_classification/yes_no.h"

namespace xt::datasets
{
    // ---------------------- YesNo ---------------------- //

    YesNo::YesNo(const std::string& root): YesNo::YesNo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    YesNo::YesNo(const std::string& root, xt::datasets::DataMode mode): YesNo::YesNo(
        root, mode, false, nullptr, nullptr)
    {
    }

    YesNo::YesNo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        YesNo::YesNo(
            root, mode, download, nullptr, nullptr)
    {
    }

    YesNo::YesNo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : YesNo::YesNo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    YesNo::YesNo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void YesNo::load_data()
    {

    }

    void YesNo::check_resources()
    {

    }
}
