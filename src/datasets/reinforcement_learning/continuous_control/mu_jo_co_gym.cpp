#include "datasets/reinforcement_learning/continuous_control/mu_jo_co_gym.h"

namespace xt::data::datasets
{
    // ---------------------- MuJoCo ---------------------- //

    MuJoCo::MuJoCo(const std::string& root): MuJoCo::MuJoCo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MuJoCo::MuJoCo(const std::string& root, xt::datasets::DataMode mode): MuJoCo::MuJoCo(
        root, mode, false, nullptr, nullptr)
    {
    }

    MuJoCo::MuJoCo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MuJoCo::MuJoCo(
            root, mode, download, nullptr, nullptr)
    {
    }

    MuJoCo::MuJoCo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MuJoCo::MuJoCo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MuJoCo::MuJoCo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MuJoCo::load_data()
    {

    }

    void MuJoCo::check_resources()
    {

    }
}
