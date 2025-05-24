#include "include/datasets/computer_vision/optical_flow_estimation/flying_chairs.h"

namespace xt::datasets
{
    // ---------------------- FlyingChairs ---------------------- //

    FlyingChairs::FlyingChairs(const std::string& root): FlyingChairs::FlyingChairs(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FlyingChairs::FlyingChairs(const std::string& root, xt::datasets::DataMode mode): FlyingChairs::FlyingChairs(
        root, mode, false, nullptr, nullptr)
    {
    }

    FlyingChairs::FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FlyingChairs::FlyingChairs(
            root, mode, download, nullptr, nullptr)
    {
    }

    FlyingChairs::FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FlyingChairs::FlyingChairs(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FlyingChairs::FlyingChairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FlyingChairs::load_data()
    {

    }

    void FlyingChairs::check_resources()
    {

    }
}
