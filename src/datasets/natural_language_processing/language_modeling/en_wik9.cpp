#include "include/datasets/natural_language_processing/language_modeling/en_wik9.h"

namespace xt::datasets
{
    // ---------------------- EnWik9 ---------------------- //

    EnWik9::EnWik9(const std::string& root): EnWik9::EnWik9(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    EnWik9::EnWik9(const std::string& root, xt::datasets::DataMode mode): EnWik9::EnWik9(
        root, mode, false, nullptr, nullptr)
    {
    }

    EnWik9::EnWik9(const std::string& root, xt::datasets::DataMode mode, bool download) :
        EnWik9::EnWik9(
            root, mode, download, nullptr, nullptr)
    {
    }

    EnWik9::EnWik9(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : EnWik9::EnWik9(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    EnWik9::EnWik9(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void EnWik9::load_data()
    {

    }

    void EnWik9::check_resources()
    {

    }
}
