#include "include/datasets/computer_vision/image_classification/places.h"

namespace xt::data::datasets
{
    // ---------------------- Places365 ---------------------- //

    Places365::Places365(const std::string& root): Places365::Places365(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Places365::Places365(const std::string& root, xt::datasets::DataMode mode): Places365::Places365(
        root, mode, false, nullptr, nullptr)
    {
    }

    Places365::Places365(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Places365::Places365(
            root, mode, download, nullptr, nullptr)
    {
    }

    Places365::Places365(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Places365::Places365(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Places365::Places365(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Places365::load_data()
    {

    }

    void Places365::check_resources()
    {

    }
}
