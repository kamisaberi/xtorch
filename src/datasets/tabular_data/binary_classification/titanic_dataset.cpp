#include "datasets/tabular_data/binary_classification/titanic_dataset.h"

namespace xt::data::datasets
{
    // ---------------------- TitanicDataset ---------------------- //

    TitanicDataset::TitanicDataset(const std::string& root): TitanicDataset::TitanicDataset(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    TitanicDataset::TitanicDataset(const std::string& root, xt::datasets::DataMode mode): TitanicDataset::TitanicDataset(
        root, mode, false, nullptr, nullptr)
    {
    }

    TitanicDataset::TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download) :
        TitanicDataset::TitanicDataset(
            root, mode, download, nullptr, nullptr)
    {
    }

    TitanicDataset::TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : TitanicDataset::TitanicDataset(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    TitanicDataset::TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void TitanicDataset::load_data()
    {

    }

    void TitanicDataset::check_resources()
    {

    }
}
