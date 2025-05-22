#include "datasets/computer_vision/stereo_matching/kitti_2015_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- Kitti2015Stereo ---------------------- //

    Kitti2015Stereo::Kitti2015Stereo(const std::string& root): Kitti2015Stereo::Kitti2015Stereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string& root, xt::datasets::DataMode mode): Kitti2015Stereo::Kitti2015Stereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Kitti2015Stereo::Kitti2015Stereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Kitti2015Stereo::Kitti2015Stereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Kitti2015Stereo::Kitti2015Stereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Kitti2015Stereo::load_data()
    {

    }

    void Kitti2015Stereo::check_resources()
    {

    }
}
