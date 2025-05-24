#include "include/datasets/computer_vision/3d_shape_generation/shapenet.h"

namespace xt::data::datasets
{
    // ---------------------- ShapeNet ---------------------- //

    ShapeNet::ShapeNet(const std::string& root): ShapeNet::ShapeNet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ShapeNet::ShapeNet(const std::string& root, xt::datasets::DataMode mode): ShapeNet::ShapeNet(
        root, mode, false, nullptr, nullptr)
    {
    }

    ShapeNet::ShapeNet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ShapeNet::ShapeNet(
            root, mode, download, nullptr, nullptr)
    {
    }

    ShapeNet::ShapeNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ShapeNet::ShapeNet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ShapeNet::ShapeNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ShapeNet::load_data()
    {

    }

    void ShapeNet::check_resources()
    {

    }
}
