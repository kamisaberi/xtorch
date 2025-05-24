#include "include/datasets/natural_language_processing/text_classification/mnli.h"


namespace xt::data::datasets
{
    // ---------------------- MNLI ---------------------- //

    MNLI::MNLI(const std::string& root): MNLI::MNLI(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MNLI::MNLI(const std::string& root, xt::datasets::DataMode mode): MNLI::MNLI(
        root, mode, false, nullptr, nullptr)
    {
    }

    MNLI::MNLI(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MNLI::MNLI(
            root, mode, download, nullptr, nullptr)
    {
    }

    MNLI::MNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MNLI::MNLI(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MNLI::MNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MNLI::load_data()
    {

    }

    void MNLI::check_resources()
    {

    }
}
