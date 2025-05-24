#include "include/datasets/biomedical_data/cancer_genomics_classification/tcga.h"

namespace xt::datasets
{
    // ---------------------- TCGA ---------------------- //

    TCGA::TCGA(const std::string& root): TCGA::TCGA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    TCGA::TCGA(const std::string& root, xt::datasets::DataMode mode): TCGA::TCGA(
        root, mode, false, nullptr, nullptr)
    {
    }

    TCGA::TCGA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        TCGA::TCGA(
            root, mode, download, nullptr, nullptr)
    {
    }

    TCGA::TCGA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : TCGA::TCGA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    TCGA::TCGA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void TCGA::load_data()
    {

    }

    void TCGA::check_resources()
    {

    }
}
