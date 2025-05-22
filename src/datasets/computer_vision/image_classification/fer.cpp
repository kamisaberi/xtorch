#include "datasets/computer_vision/image_classification/fer.h"

namespace xt::data::datasets
{
    // ---------------------- FER2013 ---------------------- //

    FER2013::FER2013(const std::string& root): FER2013::FER2013(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FER2013::FER2013(const std::string& root, xt::datasets::DataMode mode): FER2013::FER2013(
        root, mode, false, nullptr, nullptr)
    {
    }

    FER2013::FER2013(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FER2013::FER2013(
            root, mode, download, nullptr, nullptr)
    {
    }

    FER2013::FER2013(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FER2013::FER2013(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FER2013::FER2013(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FER2013::load_data()
    {

    }

    void FER2013::check_resources()
    {

    }
}
