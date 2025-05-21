#include "datasets/audio_processing/emotion_recognition/iemocap.h"

namespace xt::data::datasets
{

    IEMOCAP::IEMOCAP(const std::string& root): IEMOCAP::IEMOCAP(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    IEMOCAP::IEMOCAP(const std::string& root, xt::datasets::DataMode mode): IEMOCAP::IEMOCAP(
        root, mode, false, nullptr, nullptr)
    {
    }

    IEMOCAP::IEMOCAP(const std::string& root, xt::datasets::DataMode mode, bool download) :
        IEMOCAP::IEMOCAP(
            root, mode, download, nullptr, nullptr)
    {
    }

    IEMOCAP::IEMOCAP(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : IEMOCAP::IEMOCAP(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    IEMOCAP::IEMOCAP(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Caltech101::load_data()
    {

    }

    void Caltech101::check_resources()
    {

    }
}
