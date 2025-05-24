#include "include/datasets/natural_language_processing/text_classification/snli.h"


namespace xt::data::datasets
{
    // ---------------------- SNLI ---------------------- //

    SNLI::SNLI(const std::string& root): SNLI::SNLI(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SNLI::SNLI(const std::string& root, xt::datasets::DataMode mode): SNLI::SNLI(
        root, mode, false, nullptr, nullptr)
    {
    }

    SNLI::SNLI(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SNLI::SNLI(
            root, mode, download, nullptr, nullptr)
    {
    }

    SNLI::SNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SNLI::SNLI(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SNLI::SNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SNLI::load_data()
    {

    }

    void SNLI::check_resources()
    {

    }
}
