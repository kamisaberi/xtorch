#include "include/datasets/computer_vision/video_classification//ucf.h"

namespace xt::data::datasets
{
    // ---------------------- UCF101 ---------------------- //

    UCF101::UCF101(const std::string& root): UCF101::UCF101(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    UCF101::UCF101(const std::string& root, xt::datasets::DataMode mode): UCF101::UCF101(
        root, mode, false, nullptr, nullptr)
    {
    }

    UCF101::UCF101(const std::string& root, xt::datasets::DataMode mode, bool download) :
        UCF101::UCF101(
            root, mode, download, nullptr, nullptr)
    {
    }

    UCF101::UCF101(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : UCF101::UCF101(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    UCF101::UCF101(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void UCF101::load_data()
    {

    }

    void UCF101::check_resources()
    {

    }
}
