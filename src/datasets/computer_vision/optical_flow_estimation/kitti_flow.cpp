#include "datasets/computer_vision/optical_flow_estimation/kitti_flow.h"

namespace xt::data::datasets
{
    // ---------------------- KittiFlow ---------------------- //

    KittiFlow::KittiFlow(const std::string& root): KittiFlow::KittiFlow(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    KittiFlow::KittiFlow(const std::string& root, xt::datasets::DataMode mode): KittiFlow::KittiFlow(
        root, mode, false, nullptr, nullptr)
    {
    }

    KittiFlow::KittiFlow(const std::string& root, xt::datasets::DataMode mode, bool download) :
        KittiFlow::KittiFlow(
            root, mode, download, nullptr, nullptr)
    {
    }

    KittiFlow::KittiFlow(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : KittiFlow::KittiFlow(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    KittiFlow::KittiFlow(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void KittiFlow::load_data()
    {

    }

    void KittiFlow::check_resources()
    {

    }
}
