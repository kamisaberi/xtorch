#include "include/datasets/computer_vision/image_classification/euro_sat.h"

namespace xt::data::datasets
{
    // ---------------------- EuroSAT ---------------------- //

    EuroSAT::EuroSAT(const std::string& root): EuroSAT::EuroSAT(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    EuroSAT::EuroSAT(const std::string& root, xt::datasets::DataMode mode): EuroSAT::EuroSAT(
        root, mode, false, nullptr, nullptr)
    {
    }

    EuroSAT::EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download) :
        EuroSAT::EuroSAT(
            root, mode, download, nullptr, nullptr)
    {
    }

    EuroSAT::EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer) : EuroSAT::EuroSAT(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    EuroSAT::EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();
    }


    void EuroSAT::load_data()
    {
    }

    void EuroSAT::check_resources()
    {
    }
}
