#include "datasets/time_series/time_series_forecasting/m4_competition.h"

namespace xt::data::datasets
{
    // ---------------------- M4Competition ---------------------- //

    M4Competition::M4Competition(const std::string& root): M4Competition::M4Competition(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    M4Competition::M4Competition(const std::string& root, xt::datasets::DataMode mode): M4Competition::M4Competition(
        root, mode, false, nullptr, nullptr)
    {
    }

    M4Competition::M4Competition(const std::string& root, xt::datasets::DataMode mode, bool download) :
        M4Competition::M4Competition(
            root, mode, download, nullptr, nullptr)
    {
    }

    M4Competition::M4Competition(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : M4Competition::M4Competition(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    M4Competition::M4Competition(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void M4Competition::load_data()
    {

    }

    void M4Competition::check_resources()
    {

    }
}
