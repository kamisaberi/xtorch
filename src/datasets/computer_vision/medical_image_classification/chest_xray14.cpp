#include "datasets/computer_vision/medical_image_classification/chest_xray14.h"

namespace xt::data::datasets
{
    // ---------------------- ChestXray14 ---------------------- //

    ChestXray14::ChestXray14(const std::string& root): ChestXray14::ChestXray14(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ChestXray14::ChestXray14(const std::string& root, xt::datasets::DataMode mode): ChestXray14::ChestXray14(
        root, mode, false, nullptr, nullptr)
    {
    }

    ChestXray14::ChestXray14(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ChestXray14::ChestXray14(
            root, mode, download, nullptr, nullptr)
    {
    }

    ChestXray14::ChestXray14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ChestXray14::ChestXray14(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ChestXray14::ChestXray14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ChestXray14::load_data()
    {

    }

    void ChestXray14::check_resources()
    {

    }
}
