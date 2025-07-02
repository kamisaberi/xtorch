#include "include/datasets/computer_vision/object_detection/kitti.h"


namespace xt::datasets
{
    // ---------------------- Kitti ---------------------- //

    Kitti::Kitti(const std::string& root): Kitti::Kitti(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Kitti::Kitti(const std::string& root, xt::datasets::DataMode mode): Kitti::Kitti(
        root, mode, false, nullptr, nullptr)
    {
    }

    Kitti::Kitti(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Kitti::Kitti(
            root, mode, download, nullptr, nullptr)
    {
    }

    Kitti::Kitti(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Kitti::Kitti(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Kitti::Kitti(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Kitti::load_data()
    {

    }

    void Kitti::check_resources()
    {

    }
}
