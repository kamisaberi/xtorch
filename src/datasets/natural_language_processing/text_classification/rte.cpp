#include "datasets/natural_language_processing/text_classification/rte.h"


namespace xt::data::datasets
{
    // ---------------------- RTE ---------------------- //

    RTE::RTE(const std::string& root): RTE::RTE(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    RTE::RTE(const std::string& root, xt::datasets::DataMode mode): RTE::RTE(
        root, mode, false, nullptr, nullptr)
    {
    }

    RTE::RTE(const std::string& root, xt::datasets::DataMode mode, bool download) :
        RTE::RTE(
            root, mode, download, nullptr, nullptr)
    {
    }

    RTE::RTE(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : RTE::RTE(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    RTE::RTE(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void RTE::load_data()
    {

    }

    void RTE::check_resources()
    {

    }
}
