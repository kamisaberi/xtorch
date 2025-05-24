#include "include/datasets/tabular_data/binary_classification/ionosphere.h"

namespace xt::data::datasets
{
    // ---------------------- Ionosphere ---------------------- //

    Ionosphere::Ionosphere(const std::string& root): Ionosphere::Ionosphere(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Ionosphere::Ionosphere(const std::string& root, xt::datasets::DataMode mode): Ionosphere::Ionosphere(
        root, mode, false, nullptr, nullptr)
    {
    }

    Ionosphere::Ionosphere(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Ionosphere::Ionosphere(
            root, mode, download, nullptr, nullptr)
    {
    }

    Ionosphere::Ionosphere(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Ionosphere::Ionosphere(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Ionosphere::Ionosphere(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Ionosphere::load_data()
    {

    }

    void Ionosphere::check_resources()
    {

    }
}
