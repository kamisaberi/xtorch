#include "datasets/tabular_data/classification/iris.h"

namespace xt::data::datasets
{
    // ---------------------- Iris ---------------------- //

    Iris::Iris(const std::string& root): Iris::Iris(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Iris::Iris(const std::string& root, xt::datasets::DataMode mode): Iris::Iris(
        root, mode, false, nullptr, nullptr)
    {
    }

    Iris::Iris(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Iris::Iris(
            root, mode, download, nullptr, nullptr)
    {
    }

    Iris::Iris(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Iris::Iris(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Iris::Iris(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Iris::load_data()
    {

    }

    void Iris::check_resources()
    {

    }
}
