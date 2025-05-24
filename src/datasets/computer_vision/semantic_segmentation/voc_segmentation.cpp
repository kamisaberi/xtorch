#include "include/datasets/computer_vision/semantic_segmentation/voc_segmentation.h"


namespace xt::datasets
{
    // ---------------------- VOCSegmentation ---------------------- //

    VOCSegmentation::VOCSegmentation(const std::string& root): VOCSegmentation::VOCSegmentation(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VOCSegmentation::VOCSegmentation(const std::string& root, xt::datasets::DataMode mode): VOCSegmentation::VOCSegmentation(
        root, mode, false, nullptr, nullptr)
    {
    }

    VOCSegmentation::VOCSegmentation(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VOCSegmentation::VOCSegmentation(
            root, mode, download, nullptr, nullptr)
    {
    }

    VOCSegmentation::VOCSegmentation(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VOCSegmentation::VOCSegmentation(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VOCSegmentation::VOCSegmentation(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VOCSegmentation::load_data()
    {

    }

    void VOCSegmentation::check_resources()
    {

    }
}
