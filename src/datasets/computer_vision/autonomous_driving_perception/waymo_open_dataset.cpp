#include "include/datasets/computer_vision/autonomous_driving_perception/waymo_open_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- WaymoOpenDataset ---------------------- //

    WaymoOpenDataset::WaymoOpenDataset(const std::string& root): WaymoOpenDataset::WaymoOpenDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WaymoOpenDataset::WaymoOpenDataset(const std::string& root, xt::datasets::DataMode mode): WaymoOpenDataset::WaymoOpenDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    WaymoOpenDataset::WaymoOpenDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WaymoOpenDataset::WaymoOpenDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    WaymoOpenDataset::WaymoOpenDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WaymoOpenDataset::WaymoOpenDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WaymoOpenDataset::WaymoOpenDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WaymoOpenDataset::load_data()
    {

    }

    void WaymoOpenDataset::check_resources()
    {

    }
}
