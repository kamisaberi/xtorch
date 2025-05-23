#include "datasets/tabular_data/binary_classification/adult_census_income.h"

namespace xt::data::datasets
{
    // ---------------------- AdultCensusIncome ---------------------- //

    AdultCensusIncome::AdultCensusIncome(const std::string& root): AdultCensusIncome::AdultCensusIncome(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AdultCensusIncome::AdultCensusIncome(const std::string& root, xt::datasets::DataMode mode): AdultCensusIncome::AdultCensusIncome(
        root, mode, false, nullptr, nullptr)
    {
    }

    AdultCensusIncome::AdultCensusIncome(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AdultCensusIncome::AdultCensusIncome(
            root, mode, download, nullptr, nullptr)
    {
    }

    AdultCensusIncome::AdultCensusIncome(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AdultCensusIncome::AdultCensusIncome(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AdultCensusIncome::AdultCensusIncome(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AdultCensusIncome::load_data()
    {

    }

    void AdultCensusIncome::check_resources()
    {

    }
}
