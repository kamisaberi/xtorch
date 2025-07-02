#include "include/datasets/natural_language_processing/sequence_tagging/udpos.h"


namespace xt::datasets
{
    // ---------------------- UDPOS ---------------------- //

    UDPOS::UDPOS(const std::string& root): UDPOS::UDPOS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    UDPOS::UDPOS(const std::string& root, xt::datasets::DataMode mode): UDPOS::UDPOS(
        root, mode, false, nullptr, nullptr)
    {
    }

    UDPOS::UDPOS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        UDPOS::UDPOS(
            root, mode, download, nullptr, nullptr)
    {
    }

    UDPOS::UDPOS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : UDPOS::UDPOS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    UDPOS::UDPOS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void UDPOS::load_data()
    {

    }

    void UDPOS::check_resources()
    {

    }
}
