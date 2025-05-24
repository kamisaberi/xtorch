#include "include/datasets/computer_vision/image_classification/country.h"

namespace xt::datasets {

    // ---------------------- Country211 ---------------------- //

    Country211::Country211(const std::string& root): Country211::Country211(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Country211::Country211(const std::string& root, xt::datasets::DataMode mode): Country211::Country211(
        root, mode, false, nullptr, nullptr)
    {
    }

    Country211::Country211(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Country211::Country211(
            root, mode, download, nullptr, nullptr)
    {
    }

    Country211::Country211(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Country211::Country211(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Country211::Country211(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Country211::load_data()
    {

    }

    void Country211::check_resources()
    {

    }

}
