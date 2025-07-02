#include "include/datasets/biomedical_data/alzheimers_classification/adni.h"

namespace xt::datasets
{
    // ---------------------- ADNI ---------------------- //

    ADNI::ADNI(const std::string& root): ADNI::ADNI(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ADNI::ADNI(const std::string& root, xt::datasets::DataMode mode): ADNI::ADNI(
        root, mode, false, nullptr, nullptr)
    {
    }

    ADNI::ADNI(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ADNI::ADNI(
            root, mode, download, nullptr, nullptr)
    {
    }

    ADNI::ADNI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ADNI::ADNI(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ADNI::ADNI(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ADNI::load_data()
    {

    }

    void ADNI::check_resources()
    {

    }
}
