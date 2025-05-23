#include "datasets/tabular_data/binary_classification/banknote_authentication.h"

namespace xt::data::datasets
{
    // ---------------------- BanknoteAuthentication ---------------------- //

    BanknoteAuthentication::BanknoteAuthentication(const std::string& root): BanknoteAuthentication::BanknoteAuthentication(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    BanknoteAuthentication::BanknoteAuthentication(const std::string& root, xt::datasets::DataMode mode): BanknoteAuthentication::BanknoteAuthentication(
        root, mode, false, nullptr, nullptr)
    {
    }

    BanknoteAuthentication::BanknoteAuthentication(const std::string& root, xt::datasets::DataMode mode, bool download) :
        BanknoteAuthentication::BanknoteAuthentication(
            root, mode, download, nullptr, nullptr)
    {
    }

    BanknoteAuthentication::BanknoteAuthentication(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : BanknoteAuthentication::BanknoteAuthentication(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    BanknoteAuthentication::BanknoteAuthentication(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void BanknoteAuthentication::load_data()
    {

    }

    void BanknoteAuthentication::check_resources()
    {

    }
}
