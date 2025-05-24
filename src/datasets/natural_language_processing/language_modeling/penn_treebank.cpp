#include "include/datasets/natural_language_processing/language_modeling/penn_treebank.h"


namespace xt::datasets
{
    // ---------------------- PennTreebank ---------------------- //

    PennTreebank::PennTreebank(const std::string& root): PennTreebank::PennTreebank(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PennTreebank::PennTreebank(const std::string& root, xt::datasets::DataMode mode): PennTreebank::PennTreebank(
        root, mode, false, nullptr, nullptr)
    {
    }

    PennTreebank::PennTreebank(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PennTreebank::PennTreebank(
            root, mode, download, nullptr, nullptr)
    {
    }

    PennTreebank::PennTreebank(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PennTreebank::PennTreebank(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PennTreebank::PennTreebank(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PennTreebank::load_data()
    {

    }

    void PennTreebank::check_resources()
    {

    }
}
