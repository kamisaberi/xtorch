#include "include/datasets/tabular_data/regression/boston_housing.h"

namespace xt::data::datasets
{
    // ---------------------- BostonHousing ---------------------- //

    BostonHousing::BostonHousing(const std::string& root): BostonHousing::BostonHousing(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    BostonHousing::BostonHousing(const std::string& root, xt::datasets::DataMode mode): BostonHousing::BostonHousing(
        root, mode, false, nullptr, nullptr)
    {
    }

    BostonHousing::BostonHousing(const std::string& root, xt::datasets::DataMode mode, bool download) :
        BostonHousing::BostonHousing(
            root, mode, download, nullptr, nullptr)
    {
    }

    BostonHousing::BostonHousing(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : BostonHousing::BostonHousing(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    BostonHousing::BostonHousing(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void BostonHousing::load_data()
    {

    }

    void BostonHousing::check_resources()
    {

    }
}
