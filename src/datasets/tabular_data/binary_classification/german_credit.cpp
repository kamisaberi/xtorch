#include "include/datasets/tabular_data/binary_classification/german_credit.h"

namespace xt::data::datasets
{
    // ---------------------- GermanCredit ---------------------- //

    GermanCredit::GermanCredit(const std::string& root): GermanCredit::GermanCredit(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    GermanCredit::GermanCredit(const std::string& root, xt::datasets::DataMode mode): GermanCredit::GermanCredit(
        root, mode, false, nullptr, nullptr)
    {
    }

    GermanCredit::GermanCredit(const std::string& root, xt::datasets::DataMode mode, bool download) :
        GermanCredit::GermanCredit(
            root, mode, download, nullptr, nullptr)
    {
    }

    GermanCredit::GermanCredit(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : GermanCredit::GermanCredit(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    GermanCredit::GermanCredit(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void GermanCredit::load_data()
    {

    }

    void GermanCredit::check_resources()
    {

    }
}
