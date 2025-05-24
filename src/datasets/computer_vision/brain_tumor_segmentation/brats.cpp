#include "include/datasets/computer_vision/brain_tumor_segmentation/brats.h"

namespace xt::data::datasets
{
    // ---------------------- BraTS ---------------------- //

    BraTS::BraTS(const std::string& root): BraTS::BraTS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    BraTS::BraTS(const std::string& root, xt::datasets::DataMode mode): BraTS::BraTS(
        root, mode, false, nullptr, nullptr)
    {
    }

    BraTS::BraTS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        BraTS::BraTS(
            root, mode, download, nullptr, nullptr)
    {
    }

    BraTS::BraTS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : BraTS::BraTS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    BraTS::BraTS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void BraTS::load_data()
    {

    }

    void BraTS::check_resources()
    {

    }
}
