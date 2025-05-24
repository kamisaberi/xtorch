#include "include/datasets/natural_language_processing/unsupervised_learning/cc100.h"

namespace xt::data::datasets
{
    // ---------------------- CC100 ---------------------- //

    CC100::CC100(const std::string& root): CC100::CC100(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CC100::CC100(const std::string& root, xt::datasets::DataMode mode): CC100::CC100(
        root, mode, false, nullptr, nullptr)
    {
    }

    CC100::CC100(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CC100::CC100(
            root, mode, download, nullptr, nullptr)
    {
    }

    CC100::CC100(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CC100::CC100(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CC100::CC100(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CC100::load_data()
    {

    }

    void CC100::check_resources()
    {

    }
}
