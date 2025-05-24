#include "include/datasets/tabular_data/classification/thyroid_disease.h"

namespace xt::data::datasets
{
    // ---------------------- ThyroidDisease ---------------------- //

    ThyroidDisease::ThyroidDisease(const std::string& root): ThyroidDisease::ThyroidDisease(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ThyroidDisease::ThyroidDisease(const std::string& root, xt::datasets::DataMode mode): ThyroidDisease::ThyroidDisease(
        root, mode, false, nullptr, nullptr)
    {
    }

    ThyroidDisease::ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ThyroidDisease::ThyroidDisease(
            root, mode, download, nullptr, nullptr)
    {
    }

    ThyroidDisease::ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ThyroidDisease::ThyroidDisease(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ThyroidDisease::ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ThyroidDisease::load_data()
    {

    }

    void ThyroidDisease::check_resources()
    {

    }
}
