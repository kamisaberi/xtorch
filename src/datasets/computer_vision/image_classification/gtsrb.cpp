#include "include/datasets/computer_vision/image_classification/gtsrb.h"

namespace xt::datasets
{
    // ---------------------- GTSRB ---------------------- //

    GTSRB::GTSRB(const std::string& root): GTSRB::GTSRB(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    GTSRB::GTSRB(const std::string& root, xt::datasets::DataMode mode): GTSRB::GTSRB(
        root, mode, false, nullptr, nullptr)
    {
    }

    GTSRB::GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download) :
        GTSRB::GTSRB(
            root, mode, download, nullptr, nullptr)
    {
    }

    GTSRB::GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : GTSRB::GTSRB(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    GTSRB::GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void GTSRB::load_data()
    {

    }

    void GTSRB::check_resources()
    {

    }
}
