#include "include/datasets/computer_vision/video_classification/kinetics.h"

namespace xt::datasets
{
    // ---------------------- Kinetics ---------------------- //

    Kinetics::Kinetics(const std::string& root): Kinetics::Kinetics(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Kinetics::Kinetics(const std::string& root, xt::datasets::DataMode mode): Kinetics::Kinetics(
        root, mode, false, nullptr, nullptr)
    {
    }

    Kinetics::Kinetics(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Kinetics::Kinetics(
            root, mode, download, nullptr, nullptr)
    {
    }

    Kinetics::Kinetics(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Kinetics::Kinetics(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Kinetics::Kinetics(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Kinetics::load_data()
    {

    }

    void Kinetics::check_resources()
    {

    }
}
