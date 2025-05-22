#include "datasets/computer_vision/image_classification/flickr_8k.h"

namespace xt::data::datasets
{
    // ---------------------- Flickr8k ---------------------- //

    Flickr8k::Flickr8k(const std::string& root): Flickr8k::Flickr8k(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Flickr8k::Flickr8k(const std::string& root, xt::datasets::DataMode mode): Flickr8k::Flickr8k(
        root, mode, false, nullptr, nullptr)
    {
    }

    Flickr8k::Flickr8k(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Flickr8k::Flickr8k(
            root, mode, download, nullptr, nullptr)
    {
    }

    Flickr8k::Flickr8k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Flickr8k::Flickr8k(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Flickr8k::Flickr8k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Flickr8k::load_data()
    {

    }

    void Flickr8k::check_resources()
    {

    }
}
