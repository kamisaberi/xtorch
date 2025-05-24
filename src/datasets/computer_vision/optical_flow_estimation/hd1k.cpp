#include "include/datasets/computer_vision/optical_flow_estimation/hd1k.h"

namespace xt::data::datasets
{
    // ---------------------- HD1K ---------------------- //

    HD1K::HD1K(const std::string& root): HD1K::HD1K(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    HD1K::HD1K(const std::string& root, xt::datasets::DataMode mode): HD1K::HD1K(
        root, mode, false, nullptr, nullptr)
    {
    }

    HD1K::HD1K(const std::string& root, xt::datasets::DataMode mode, bool download) :
        HD1K::HD1K(
            root, mode, download, nullptr, nullptr)
    {
    }

    HD1K::HD1K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : HD1K::HD1K(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    HD1K::HD1K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void HD1K::load_data()
    {

    }

    void HD1K::check_resources()
    {

    }
}
