#include "datasets/tabular_data/classification/ecoli.h"

namespace xt::data::datasets
{
    // ---------------------- Ecoli ---------------------- //

    Ecoli::Ecoli(const std::string& root): Ecoli::Ecoli(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Ecoli::Ecoli(const std::string& root, xt::datasets::DataMode mode): Ecoli::Ecoli(
        root, mode, false, nullptr, nullptr)
    {
    }

    Ecoli::Ecoli(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Ecoli::Ecoli(
            root, mode, download, nullptr, nullptr)
    {
    }

    Ecoli::Ecoli(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Ecoli::Ecoli(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Ecoli::Ecoli(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Ecoli::load_data()
    {

    }

    void Ecoli::check_resources()
    {

    }
}
