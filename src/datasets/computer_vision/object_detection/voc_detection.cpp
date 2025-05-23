#include "datasets/computer_vision/object_detection/voc_detection.h"


namespace xt::data::datasets
{
    // ---------------------- VOCDetection ---------------------- //

    VOCDetection::VOCDetection(const std::string& root): VOCDetection::VOCDetection(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VOCDetection::VOCDetection(const std::string& root, xt::datasets::DataMode mode): VOCDetection::VOCDetection(
        root, mode, false, nullptr, nullptr)
    {
    }

    VOCDetection::VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VOCDetection::VOCDetection(
            root, mode, download, nullptr, nullptr)
    {
    }

    VOCDetection::VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VOCDetection::VOCDetection(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VOCDetection::VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VOCDetection::load_data()
    {

    }

    void VOCDetection::check_resources()
    {

    }
}
