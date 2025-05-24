#include "datasets/computer_vision/3d_point_cloud_analysis/model_net40.h"

namespace xt::data::datasets
{
    // ---------------------- WheatSeeds ---------------------- //

    WheatSeeds::WheatSeeds(const std::string& root): WheatSeeds::WheatSeeds(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WheatSeeds::WheatSeeds(const std::string& root, xt::datasets::DataMode mode): WheatSeeds::WheatSeeds(
        root, mode, false, nullptr, nullptr)
    {
    }

    WheatSeeds::WheatSeeds(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WheatSeeds::WheatSeeds(
            root, mode, download, nullptr, nullptr)
    {
    }

    WheatSeeds::WheatSeeds(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WheatSeeds::WheatSeeds(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WheatSeeds::WheatSeeds(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WheatSeeds::load_data()
    {

    }

    void WheatSeeds::check_resources()
    {

    }
}
