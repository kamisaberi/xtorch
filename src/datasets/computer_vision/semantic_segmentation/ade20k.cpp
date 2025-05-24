#include "include/datasets/computer_vision/semantic_segmentation/ade20k.h"

namespace xt::datasets
{
    // ---------------------- ADE20K ---------------------- //

    ADE20K::ADE20K(const std::string& root): ADE20K::ADE20K(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ADE20K::ADE20K(const std::string& root, xt::datasets::DataMode mode): ADE20K::ADE20K(
        root, mode, false, nullptr, nullptr)
    {
    }

    ADE20K::ADE20K(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ADE20K::ADE20K(
            root, mode, download, nullptr, nullptr)
    {
    }

    ADE20K::ADE20K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ADE20K::ADE20K(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ADE20K::ADE20K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ADE20K::load_data()
    {

    }

    void ADE20K::check_resources()
    {

    }
}
