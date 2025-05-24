#include "datasets/tabular_data/classification/yeast.h"

namespace xt::data::datasets
{
    // ---------------------- Yeast ---------------------- //

    Yeast::Yeast(const std::string& root): Yeast::Yeast(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Yeast::Yeast(const std::string& root, xt::datasets::DataMode mode): Yeast::Yeast(
        root, mode, false, nullptr, nullptr)
    {
    }

    Yeast::Yeast(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Yeast::Yeast(
            root, mode, download, nullptr, nullptr)
    {
    }

    Yeast::Yeast(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Yeast::Yeast(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Yeast::Yeast(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Yeast::load_data()
    {

    }

    void Yeast::check_resources()
    {

    }
}
