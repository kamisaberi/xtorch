#include "datasets/computer_vision/video_classification/hmdb.h"

namespace xt::data::datasets
{
    // ---------------------- HMDB51 ---------------------- //

    HMDB51::HMDB51(const std::string& root): HMDB51::HMDB51(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    HMDB51::HMDB51(const std::string& root, xt::datasets::DataMode mode): HMDB51::HMDB51(
        root, mode, false, nullptr, nullptr)
    {
    }

    HMDB51::HMDB51(const std::string& root, xt::datasets::DataMode mode, bool download) :
        HMDB51::HMDB51(
            root, mode, download, nullptr, nullptr)
    {
    }

    HMDB51::HMDB51(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : HMDB51::HMDB51(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    HMDB51::HMDB51(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void HMDB51::load_data()
    {

    }

    void HMDB51::check_resources()
    {

    }
}
