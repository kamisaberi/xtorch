#include "include/datasets/computer_vision/semantic_segmentation/cityscapes.h"


namespace xt::data::datasets
{
    // ---------------------- Cityscapes ---------------------- //

    Cityscapes::Cityscapes(const std::string& root): Cityscapes::Cityscapes(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Cityscapes::Cityscapes(const std::string& root, xt::datasets::DataMode mode): Cityscapes::Cityscapes(
        root, mode, false, nullptr, nullptr)
    {
    }

    Cityscapes::Cityscapes(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Cityscapes::Cityscapes(
            root, mode, download, nullptr, nullptr)
    {
    }

    Cityscapes::Cityscapes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Cityscapes::Cityscapes(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Cityscapes::Cityscapes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Cityscapes::load_data()
    {

    }

    void Cityscapes::check_resources()
    {

    }
}
