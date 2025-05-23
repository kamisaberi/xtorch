#include "datasets/computer_vision/image_pair_tasks/lfw_pairs.h"

namespace xt::data::datasets
{
    // ---------------------- LFWPairs ---------------------- //

    LFWPairs::LFWPairs(const std::string& root): LFWPairs::LFWPairs(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LFWPairs::LFWPairs(const std::string& root, xt::datasets::DataMode mode): LFWPairs::LFWPairs(
        root, mode, false, nullptr, nullptr)
    {
    }

    LFWPairs::LFWPairs(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LFWPairs::LFWPairs(
            root, mode, download, nullptr, nullptr)
    {
    }

    LFWPairs::LFWPairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LFWPairs::LFWPairs(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LFWPairs::LFWPairs(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LFWPairs::load_data()
    {

    }

    void LFWPairs::check_resources()
    {

    }
}
