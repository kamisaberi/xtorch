#include "include/datasets/computer_vision/image_classification/flowers.h"

namespace xt::datasets
{
    // ---------------------- Flowers102 ---------------------- //

    Flowers102::Flowers102(const std::string& root): Flowers102::Flowers102(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Flowers102::Flowers102(const std::string& root, xt::datasets::DataMode mode): Flowers102::Flowers102(
        root, mode, false, nullptr, nullptr)
    {
    }

    Flowers102::Flowers102(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Flowers102::Flowers102(
            root, mode, download, nullptr, nullptr)
    {
    }

    Flowers102::Flowers102(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Flowers102::Flowers102(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Flowers102::Flowers102(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Flowers102::load_data()
    {

    }

    void Flowers102::check_resources()
    {

    }
}
