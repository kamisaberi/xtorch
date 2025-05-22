#include "datasets/computer_vision/image_classification/stanford_cars.h"

namespace xt::data::datasets
{
    // ---------------------- StanfordCars ---------------------- //

    StanfordCars::StanfordCars(const std::string& root): StanfordCars::StanfordCars(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    StanfordCars::StanfordCars(const std::string& root, xt::datasets::DataMode mode): StanfordCars::StanfordCars(
        root, mode, false, nullptr, nullptr)
    {
    }

    StanfordCars::StanfordCars(const std::string& root, xt::datasets::DataMode mode, bool download) :
        StanfordCars::StanfordCars(
            root, mode, download, nullptr, nullptr)
    {
    }

    StanfordCars::StanfordCars(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : StanfordCars::StanfordCars(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    StanfordCars::StanfordCars(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void StanfordCars::load_data()
    {

    }

    void StanfordCars::check_resources()
    {

    }
}
