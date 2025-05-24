#include "include/datasets/tabular_data/binary_classification/mushroom_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- MushroomDataset ---------------------- //

    MushroomDataset::MushroomDataset(const std::string& root): MushroomDataset::MushroomDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MushroomDataset::MushroomDataset(const std::string& root, xt::datasets::DataMode mode): MushroomDataset::MushroomDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    MushroomDataset::MushroomDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MushroomDataset::MushroomDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    MushroomDataset::MushroomDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MushroomDataset::MushroomDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MushroomDataset::MushroomDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MushroomDataset::load_data()
    {

    }

    void MushroomDataset::check_resources()
    {

    }
}
