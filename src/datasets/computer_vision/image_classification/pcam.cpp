#include "include/datasets/computer_vision/image_classification/pcam.h"

namespace xt::data::datasets
{
    // ---------------------- PCAM ---------------------- //

    PCAM::PCAM(const std::string& root): PCAM::PCAM(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PCAM::PCAM(const std::string& root, xt::datasets::DataMode mode): PCAM::PCAM(
        root, mode, false, nullptr, nullptr)
    {
    }

    PCAM::PCAM(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PCAM::PCAM(
            root, mode, download, nullptr, nullptr)
    {
    }

    PCAM::PCAM(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PCAM::PCAM(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PCAM::PCAM(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PCAM::load_data()
    {

    }

    void PCAM::check_resources()
    {

    }
}
