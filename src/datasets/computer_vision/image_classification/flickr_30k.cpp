#include "include/datasets/computer_vision/image_classification/flickr_30k.h"

namespace xt::data::datasets
{
    // ---------------------- Flickr30k ---------------------- //

    Flickr30k::Flickr30k(const std::string& root): Flickr30k::Flickr30k(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Flickr30k::Flickr30k(const std::string& root, xt::datasets::DataMode mode): Flickr30k::Flickr30k(
        root, mode, false, nullptr, nullptr)
    {
    }

    Flickr30k::Flickr30k(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Flickr30k::Flickr30k(
            root, mode, download, nullptr, nullptr)
    {
    }

    Flickr30k::Flickr30k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Flickr30k::Flickr30k(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Flickr30k::Flickr30k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Flickr30k::load_data()
    {

    }

    void Flickr30k::check_resources()
    {

    }
}
