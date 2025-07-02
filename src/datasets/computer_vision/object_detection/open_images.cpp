#include "include/datasets/computer_vision/object_detection/open_images.h"

namespace xt::datasets
{
    // ---------------------- OpenImages ---------------------- //

    OpenImages::OpenImages(const std::string& root): OpenImages::OpenImages(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    OpenImages::OpenImages(const std::string& root, xt::datasets::DataMode mode): OpenImages::OpenImages(
        root, mode, false, nullptr, nullptr)
    {
    }

    OpenImages::OpenImages(const std::string& root, xt::datasets::DataMode mode, bool download) :
        OpenImages::OpenImages(
            root, mode, download, nullptr, nullptr)
    {
    }

    OpenImages::OpenImages(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : OpenImages::OpenImages(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    OpenImages::OpenImages(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void OpenImages::load_data()
    {

    }

    void OpenImages::check_resources()
    {

    }
}
