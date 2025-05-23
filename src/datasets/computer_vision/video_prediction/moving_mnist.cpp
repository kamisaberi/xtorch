#include "datasets/computer_vision/video_prediction/moving_mnist.h"

namespace xt::data::datasets
{
    // ---------------------- MovingMNIST ---------------------- //

    MovingMNIST::MovingMNIST(const std::string& root): MovingMNIST::MovingMNIST(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MovingMNIST::MovingMNIST(const std::string& root, xt::datasets::DataMode mode): MovingMNIST::MovingMNIST(
        root, mode, false, nullptr, nullptr)
    {
    }

    MovingMNIST::MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MovingMNIST::MovingMNIST(
            root, mode, download, nullptr, nullptr)
    {
    }

    MovingMNIST::MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MovingMNIST::MovingMNIST(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MovingMNIST::MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MovingMNIST::load_data()
    {

    }

    void MovingMNIST::check_resources()
    {

    }
}
