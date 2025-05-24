#include "include/datasets/computer_vision/video_object_segmentation/davis.h"

namespace xt::data::datasets
{
    // ---------------------- DAVIS ---------------------- //

    DAVIS::DAVIS(const std::string& root): DAVIS::DAVIS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    DAVIS::DAVIS(const std::string& root, xt::datasets::DataMode mode): DAVIS::DAVIS(
        root, mode, false, nullptr, nullptr)
    {
    }

    DAVIS::DAVIS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        DAVIS::DAVIS(
            root, mode, download, nullptr, nullptr)
    {
    }

    DAVIS::DAVIS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : DAVIS::DAVIS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    DAVIS::DAVIS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void DAVIS::load_data()
    {

    }

    void DAVIS::check_resources()
    {

    }
}
