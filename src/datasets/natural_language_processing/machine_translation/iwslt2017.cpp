#include "include/datasets/natural_language_processing/machine_translation/iwslt2017.h"


namespace xt::datasets
{
    // ---------------------- IWSLT2017 ---------------------- //

    IWSLT2017::IWSLT2017(const std::string& root): IWSLT2017::IWSLT2017(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    IWSLT2017::IWSLT2017(const std::string& root, xt::datasets::DataMode mode): IWSLT2017::IWSLT2017(
        root, mode, false, nullptr, nullptr)
    {
    }

    IWSLT2017::IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download) :
        IWSLT2017::IWSLT2017(
            root, mode, download, nullptr, nullptr)
    {
    }

    IWSLT2017::IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : IWSLT2017::IWSLT2017(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    IWSLT2017::IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void IWSLT2017::load_data()
    {

    }

    void IWSLT2017::check_resources()
    {

    }
}
