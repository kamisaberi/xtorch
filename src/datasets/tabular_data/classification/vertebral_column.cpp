#include "datasets/tabular_data/classification/vertebral_column.h"

namespace xt::data::datasets
{
    // ---------------------- VertebralColumn ---------------------- //

    VertebralColumn::VertebralColumn(const std::string& root): VertebralColumn::VertebralColumn(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VertebralColumn::VertebralColumn(const std::string& root, xt::datasets::DataMode mode): VertebralColumn::VertebralColumn(
        root, mode, false, nullptr, nullptr)
    {
    }

    VertebralColumn::VertebralColumn(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VertebralColumn::VertebralColumn(
            root, mode, download, nullptr, nullptr)
    {
    }

    VertebralColumn::VertebralColumn(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VertebralColumn::VertebralColumn(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VertebralColumn::VertebralColumn(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VertebralColumn::load_data()
    {

    }

    void VertebralColumn::check_resources()
    {

    }
}
