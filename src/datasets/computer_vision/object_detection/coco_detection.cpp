#include "include/datasets/computer_vision/object_detection/coco_detection.h"


namespace xt::data::datasets
{
    // ---------------------- CocoDetection ---------------------- //

    CocoDetection::CocoDetection(const std::string& root): CocoDetection::CocoDetection(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CocoDetection::CocoDetection(const std::string& root, xt::datasets::DataMode mode): CocoDetection::CocoDetection(
        root, mode, false, nullptr, nullptr)
    {
    }

    CocoDetection::CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CocoDetection::CocoDetection(
            root, mode, download, nullptr, nullptr)
    {
    }

    CocoDetection::CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CocoDetection::CocoDetection(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CocoDetection::CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CocoDetection::load_data()
    {

    }

    void CocoDetection::check_resources()
    {

    }
}
