#include "datasets/natural_language_processing/text_classification/qnli.h"


namespace xt::data::datasets
{
    // ---------------------- QNLI ---------------------- //

    QNLI::QNLI(const std::string& root): QNLI::QNLI(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    QNLI::QNLI(const std::string& root, xt::datasets::DataMode mode): QNLI::QNLI(
        root, mode, false, nullptr, nullptr)
    {
    }

    QNLI::QNLI(const std::string& root, xt::datasets::DataMode mode, bool download) :
        QNLI::QNLI(
            root, mode, download, nullptr, nullptr)
    {
    }

    QNLI::QNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : QNLI::QNLI(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    QNLI::QNLI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void QNLI::load_data()
    {

    }

    void QNLI::check_resources()
    {

    }
}
