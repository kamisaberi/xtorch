#include "include/datasets/computer_vision/image_classification/sun.h"

namespace xt::datasets
{
    // ---------------------- SUN397 ---------------------- //

    SUN397::SUN397(const std::string& root): SUN397::SUN397(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SUN397::SUN397(const std::string& root, xt::datasets::DataMode mode): SUN397::SUN397(
        root, mode, false, nullptr, nullptr)
    {
    }

    SUN397::SUN397(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SUN397::SUN397(
            root, mode, download, nullptr, nullptr)
    {
    }

    SUN397::SUN397(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SUN397::SUN397(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SUN397::SUN397(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SUN397::load_data()
    {

    }

    void SUN397::check_resources()
    {

    }
}
