#include "include/datasets/tabular_data/binary_classification/habermans_survival.h"

namespace xt::data::datasets
{
    // ---------------------- HabermansSurvival ---------------------- //

    HabermansSurvival::HabermansSurvival(const std::string& root): HabermansSurvival::HabermansSurvival(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    HabermansSurvival::HabermansSurvival(const std::string& root, xt::datasets::DataMode mode): HabermansSurvival::HabermansSurvival(
        root, mode, false, nullptr, nullptr)
    {
    }

    HabermansSurvival::HabermansSurvival(const std::string& root, xt::datasets::DataMode mode, bool download) :
        HabermansSurvival::HabermansSurvival(
            root, mode, download, nullptr, nullptr)
    {
    }

    HabermansSurvival::HabermansSurvival(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : HabermansSurvival::HabermansSurvival(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    HabermansSurvival::HabermansSurvival(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void HabermansSurvival::load_data()
    {

    }

    void HabermansSurvival::check_resources()
    {

    }
}
