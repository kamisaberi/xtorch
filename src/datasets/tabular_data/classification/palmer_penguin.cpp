#include "include/datasets/tabular_data/classification/palmer_penguin.h"

namespace xt::datasets
{
    // ---------------------- PalmerPenguin ---------------------- //

    PalmerPenguin::PalmerPenguin(const std::string& root): PalmerPenguin::PalmerPenguin(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PalmerPenguin::PalmerPenguin(const std::string& root, xt::datasets::DataMode mode): PalmerPenguin::PalmerPenguin(
        root, mode, false, nullptr, nullptr)
    {
    }

    PalmerPenguin::PalmerPenguin(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PalmerPenguin::PalmerPenguin(
            root, mode, download, nullptr, nullptr)
    {
    }

    PalmerPenguin::PalmerPenguin(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PalmerPenguin::PalmerPenguin(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PalmerPenguin::PalmerPenguin(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PalmerPenguin::load_data()
    {

    }

    void PalmerPenguin::check_resources()
    {

    }
}
