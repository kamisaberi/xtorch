#include "datasets/natural_language_processing/machine_translation/iwslt2016.h"

namespace xt::data::datasets
{
    // ---------------------- IWSLT2016 ---------------------- //

    IWSLT2016::IWSLT2016(const std::string& root): IWSLT2016::IWSLT2016(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    IWSLT2016::IWSLT2016(const std::string& root, xt::datasets::DataMode mode): IWSLT2016::IWSLT2016(
        root, mode, false, nullptr, nullptr)
    {
    }

    IWSLT2016::IWSLT2016(const std::string& root, xt::datasets::DataMode mode, bool download) :
        IWSLT2016::IWSLT2016(
            root, mode, download, nullptr, nullptr)
    {
    }

    IWSLT2016::IWSLT2016(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : IWSLT2016::IWSLT2016(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    IWSLT2016::IWSLT2016(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void IWSLT2016::load_data()
    {

    }

    void IWSLT2016::check_resources()
    {

    }
}
