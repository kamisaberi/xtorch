#include "include/datasets/computer_vision/stereo_matching/kitti_2012_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- Kitti2012Stereo ---------------------- //

    Kitti2012Stereo::Kitti2012Stereo(const std::string& root): Kitti2012Stereo::Kitti2012Stereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode): Kitti2012Stereo::Kitti2012Stereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Kitti2012Stereo::Kitti2012Stereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Kitti2012Stereo::Kitti2012Stereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Kitti2012Stereo::Kitti2012Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Kitti2012Stereo::load_data()
    {

    }

    void Kitti2012Stereo::check_resources()
    {

    }
}
