#include "datasets/computer_vision/semantic_segmentation/sb_dataset.h"


namespace xt::data::datasets
{
    // ---------------------- SBDataset ---------------------- //

    SBDataset::SBDataset(const std::string& root): SBDataset::SBDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SBDataset::SBDataset(const std::string& root, xt::datasets::DataMode mode): SBDataset::SBDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    SBDataset::SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SBDataset::SBDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    SBDataset::SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SBDataset::SBDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SBDataset::SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SBDataset::load_data()
    {

    }

    void SBDataset::check_resources()
    {

    }
}
