#include "include/datasets/computer_vision/optical_flow_estimation/flying_things_3d.h"

namespace xt::data::datasets
{
    // ---------------------- FlyingThings3D ---------------------- //

    FlyingThings3D::FlyingThings3D(const std::string& root): FlyingThings3D::FlyingThings3D(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FlyingThings3D::FlyingThings3D(const std::string& root, xt::datasets::DataMode mode): FlyingThings3D::FlyingThings3D(
        root, mode, false, nullptr, nullptr)
    {
    }

    FlyingThings3D::FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FlyingThings3D::FlyingThings3D(
            root, mode, download, nullptr, nullptr)
    {
    }

    FlyingThings3D::FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FlyingThings3D::FlyingThings3D(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FlyingThings3D::FlyingThings3D(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FlyingThings3D::load_data()
    {

    }

    void FlyingThings3D::check_resources()
    {

    }
}
