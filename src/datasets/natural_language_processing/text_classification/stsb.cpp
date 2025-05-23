#include "datasets/natural_language_processing/text_classification/stsb.h"

namespace xt::data::datasets
{
    // ---------------------- STSB ---------------------- //

    STSB::STSB(const std::string& root): STSB::STSB(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    STSB::STSB(const std::string& root, xt::datasets::DataMode mode): STSB::STSB(
        root, mode, false, nullptr, nullptr)
    {
    }

    STSB::STSB(const std::string& root, xt::datasets::DataMode mode, bool download) :
        STSB::STSB(
            root, mode, download, nullptr, nullptr)
    {
    }

    STSB::STSB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : STSB::STSB(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    STSB::STSB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void STSB::load_data()
    {

    }

    void STSB::check_resources()
    {

    }
}
