#include "datasets/tabular_data/classification/balance_scale.h"

namespace xt::data::datasets
{
    // ---------------------- BalanceScale ---------------------- //

    BalanceScale::BalanceScale(const std::string& root): BalanceScale::BalanceScale(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    BalanceScale::BalanceScale(const std::string& root, xt::datasets::DataMode mode): BalanceScale::BalanceScale(
        root, mode, false, nullptr, nullptr)
    {
    }

    BalanceScale::BalanceScale(const std::string& root, xt::datasets::DataMode mode, bool download) :
        BalanceScale::BalanceScale(
            root, mode, download, nullptr, nullptr)
    {
    }

    BalanceScale::BalanceScale(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : BalanceScale::BalanceScale(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    BalanceScale::BalanceScale(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void BalanceScale::load_data()
    {

    }

    void BalanceScale::check_resources()
    {

    }
}
