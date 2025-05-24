#include "include/datasets/computer_vision/image_classification/usps.h"

namespace xt::datasets
{
    // ---------------------- USPS ---------------------- //

    USPS::USPS(const std::string& root): USPS::USPS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    USPS::USPS(const std::string& root, xt::datasets::DataMode mode): USPS::USPS(
        root, mode, false, nullptr, nullptr)
    {
    }

    USPS::USPS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        USPS::USPS(
            root, mode, download, nullptr, nullptr)
    {
    }

    USPS::USPS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : USPS::USPS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    USPS::USPS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void USPS::load_data()
    {

    }

    void USPS::check_resources()
    {

    }
}
