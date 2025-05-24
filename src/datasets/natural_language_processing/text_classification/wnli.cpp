#include "include/datasets/natural_language_processing/text_classification/wnli.h"


namespace xt::datasets
{
    // ---------------------- WNLI ---------------------- //

    WNLI::WNLI(const std::string& root): WNLI::WNLI(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WNLI::WNLI(const std::string& root, xt::datasets::DataMode mode): WNLI::WNLI(
        root, mode, false, nullptr, nullptr)
    {
    }

    WNLI::WNLI(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WNLI::WNLI(
            root, mode, download, nullptr, nullptr)
    {
    }

    WNLI::WNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WNLI::WNLI(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WNLI::WNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WNLI::load_data()
    {

    }

    void WNLI::check_resources()
    {

    }
}
