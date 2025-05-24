#include "include/datasets/natural_language_processing/text_classification/sst2.h"


namespace xt::data::datasets
{
    // ---------------------- SST2 ---------------------- //

    SST2::SST2(const std::string& root): SST2::SST2(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SST2::SST2(const std::string& root, xt::datasets::DataMode mode): SST2::SST2(
        root, mode, false, nullptr, nullptr)
    {
    }

    SST2::SST2(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SST2::SST2(
            root, mode, download, nullptr, nullptr)
    {
    }

    SST2::SST2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SST2::SST2(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SST2::SST2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SST2::load_data()
    {

    }

    void SST2::check_resources()
    {

    }
}
