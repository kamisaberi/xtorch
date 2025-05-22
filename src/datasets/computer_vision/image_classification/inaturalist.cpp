#include "datasets/computer_vision/image_classification/inaturalist.h"

namespace xt::data::datasets
{
    // ---------------------- INaturalist ---------------------- //

    INaturalist::INaturalist(const std::string& root): INaturalist::INaturalist(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    INaturalist::INaturalist(const std::string& root, xt::datasets::DataMode mode): INaturalist::INaturalist(
        root, mode, false, nullptr, nullptr)
    {
    }

    INaturalist::INaturalist(const std::string& root, xt::datasets::DataMode mode, bool download) :
        INaturalist::INaturalist(
            root, mode, download, nullptr, nullptr)
    {
    }

    INaturalist::INaturalist(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : INaturalist::INaturalist(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    INaturalist::INaturalist(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void INaturalist::load_data()
    {

    }

    void INaturalist::check_resources()
    {

    }
}
