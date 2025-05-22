#include "datasets/computer_vision/image_classification/imagenet.h"

namespace xt::data::datasets
{
    // ---------------------- ImageNet ---------------------- //

    ImageNet::ImageNet(const std::string& root): ImageNet::ImageNet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ImageNet::ImageNet(const std::string& root, xt::datasets::DataMode mode): ImageNet::ImageNet(
        root, mode, false, nullptr, nullptr)
    {
    }

    ImageNet::ImageNet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ImageNet::ImageNet(
            root, mode, download, nullptr, nullptr)
    {
    }

    ImageNet::ImageNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ImageNet::ImageNet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ImageNet::ImageNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ImageNet::load_data()
    {

    }

    void ImageNet::check_resources()
    {

    }
}
