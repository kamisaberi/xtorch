#include "include/datasets/computer_vision/object_detection/space_net.h"

namespace xt::data::datasets
{
    // ---------------------- SpaceNet ---------------------- //

    SpaceNet::SpaceNet(const std::string& root): SpaceNet::SpaceNet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SpaceNet::SpaceNet(const std::string& root, xt::datasets::DataMode mode): SpaceNet::SpaceNet(
        root, mode, false, nullptr, nullptr)
    {
    }

    SpaceNet::SpaceNet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SpaceNet::SpaceNet(
            root, mode, download, nullptr, nullptr)
    {
    }

    SpaceNet::SpaceNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SpaceNet::SpaceNet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SpaceNet::SpaceNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SpaceNet::load_data()
    {

    }

    void SpaceNet::check_resources()
    {

    }
}
