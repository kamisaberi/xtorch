#include "include/datasets/tabular_data/classification/zoo_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- ZooDataset ---------------------- //

    ZooDataset::ZooDataset(const std::string& root): ZooDataset::ZooDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ZooDataset::ZooDataset(const std::string& root, xt::datasets::DataMode mode): ZooDataset::ZooDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    ZooDataset::ZooDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ZooDataset::ZooDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    ZooDataset::ZooDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ZooDataset::ZooDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ZooDataset::ZooDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ZooDataset::load_data()
    {

    }

    void ZooDataset::check_resources()
    {

    }
}
