#include "datasets/tabular_data/regression_classification/abalone.h"

namespace xt::data::datasets
{
    // ---------------------- Abalone ---------------------- //

    Abalone::Abalone(const std::string& root): Abalone::Abalone(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Abalone::Abalone(const std::string& root, xt::datasets::DataMode mode): Abalone::Abalone(
        root, mode, false, nullptr, nullptr)
    {
    }

    Abalone::Abalone(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Abalone::Abalone(
            root, mode, download, nullptr, nullptr)
    {
    }

    Abalone::Abalone(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Abalone::Abalone(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Abalone::Abalone(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Abalone::load_data()
    {

    }

    void Abalone::check_resources()
    {

    }
}
