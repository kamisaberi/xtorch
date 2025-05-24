#include "include/datasets/computer_vision/anomaly_detection/mvtec_ad.h"

namespace xt::data::datasets
{
    // ---------------------- MVTecAD ---------------------- //

    MVTecAD::MVTecAD(const std::string& root): MVTecAD::MVTecAD(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MVTecAD::MVTecAD(const std::string& root, xt::datasets::DataMode mode): MVTecAD::MVTecAD(
        root, mode, false, nullptr, nullptr)
    {
    }

    MVTecAD::MVTecAD(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MVTecAD::MVTecAD(
            root, mode, download, nullptr, nullptr)
    {
    }

    MVTecAD::MVTecAD(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MVTecAD::MVTecAD(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MVTecAD::MVTecAD(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MVTecAD::load_data()
    {

    }

    void MVTecAD::check_resources()
    {

    }
}
