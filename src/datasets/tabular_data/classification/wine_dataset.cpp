#include "include/datasets/tabular_data/classification/wine_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- WineDataset ---------------------- //

    WineDataset::WineDataset(const std::string& root): WineDataset::WineDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WineDataset::WineDataset(const std::string& root, xt::datasets::DataMode mode): WineDataset::WineDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    WineDataset::WineDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WineDataset::WineDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    WineDataset::WineDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WineDataset::WineDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WineDataset::WineDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WineDataset::load_data()
    {

    }

    void WineDataset::check_resources()
    {

    }
}
