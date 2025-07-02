#include "include/datasets/time_series/time_series_forecasting/electricity_load_diagrams.h"

namespace xt::datasets
{
    // ---------------------- ElectricityLoadDiagrams ---------------------- //

    ElectricityLoadDiagrams::ElectricityLoadDiagrams(const std::string& root): ElectricityLoadDiagrams::ElectricityLoadDiagrams(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ElectricityLoadDiagrams::ElectricityLoadDiagrams(const std::string& root, xt::datasets::DataMode mode): ElectricityLoadDiagrams::ElectricityLoadDiagrams(
        root, mode, false, nullptr, nullptr)
    {
    }

    ElectricityLoadDiagrams::ElectricityLoadDiagrams(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ElectricityLoadDiagrams::ElectricityLoadDiagrams(
            root, mode, download, nullptr, nullptr)
    {
    }

    ElectricityLoadDiagrams::ElectricityLoadDiagrams(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ElectricityLoadDiagrams::ElectricityLoadDiagrams(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ElectricityLoadDiagrams::ElectricityLoadDiagrams(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ElectricityLoadDiagrams::load_data()
    {

    }

    void ElectricityLoadDiagrams::check_resources()
    {

    }
}
