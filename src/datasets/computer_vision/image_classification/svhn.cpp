#include "include/datasets/computer_vision/image_classification/svhn.h"

namespace xt::data::datasets
{
    // ---------------------- SVHN ---------------------- //

    SVHN::SVHN(const std::string& root): SVHN::SVHN(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SVHN::SVHN(const std::string& root, xt::datasets::DataMode mode): SVHN::SVHN(
        root, mode, false, nullptr, nullptr)
    {
    }

    SVHN::SVHN(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SVHN::SVHN(
            root, mode, download, nullptr, nullptr)
    {
    }

    SVHN::SVHN(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SVHN::SVHN(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SVHN::SVHN(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SVHN::load_data()
    {

    }

    void SVHN::check_resources()
    {

    }
}
