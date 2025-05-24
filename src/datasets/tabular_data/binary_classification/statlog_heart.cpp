#include "include/datasets/tabular_data/binary_classification/statlog_heart.h"

namespace xt::data::datasets
{
    // ---------------------- StatlogHeart ---------------------- //

    StatlogHeart::StatlogHeart(const std::string& root): StatlogHeart::StatlogHeart(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    StatlogHeart::StatlogHeart(const std::string& root, xt::datasets::DataMode mode): StatlogHeart::StatlogHeart(
        root, mode, false, nullptr, nullptr)
    {
    }

    StatlogHeart::StatlogHeart(const std::string& root, xt::datasets::DataMode mode, bool download) :
        StatlogHeart::StatlogHeart(
            root, mode, download, nullptr, nullptr)
    {
    }

    StatlogHeart::StatlogHeart(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : StatlogHeart::StatlogHeart(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    StatlogHeart::StatlogHeart(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void StatlogHeart::load_data()
    {

    }

    void StatlogHeart::check_resources()
    {

    }
}
