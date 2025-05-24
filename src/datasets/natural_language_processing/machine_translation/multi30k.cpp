#include "include/datasets/natural_language_processing/machine_translation/multi30k.h"


namespace xt::data::datasets
{
    // ---------------------- MULTI30k ---------------------- //

    MULTI30k::MULTI30k(const std::string& root): MULTI30k::MULTI30k(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MULTI30k::MULTI30k(const std::string& root, xt::datasets::DataMode mode): MULTI30k::MULTI30k(
        root, mode, false, nullptr, nullptr)
    {
    }

    MULTI30k::MULTI30k(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MULTI30k::MULTI30k(
            root, mode, download, nullptr, nullptr)
    {
    }

    MULTI30k::MULTI30k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MULTI30k::MULTI30k(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MULTI30k::MULTI30k(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MULTI30k::load_data()
    {

    }

    void MULTI30k::check_resources()
    {

    }
}
