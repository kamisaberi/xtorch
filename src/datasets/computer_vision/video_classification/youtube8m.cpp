#include "include/datasets/computer_vision/video_classification/youtube8m.h"

namespace xt::data::datasets
{
    // ---------------------- YouTube8M ---------------------- //

    YouTube8M::YouTube8M(const std::string& root): YouTube8M::YouTube8M(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    YouTube8M::YouTube8M(const std::string& root, xt::datasets::DataMode mode): YouTube8M::YouTube8M(
        root, mode, false, nullptr, nullptr)
    {
    }

    YouTube8M::YouTube8M(const std::string& root, xt::datasets::DataMode mode, bool download) :
        YouTube8M::YouTube8M(
            root, mode, download, nullptr, nullptr)
    {
    }

    YouTube8M::YouTube8M(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : YouTube8M::YouTube8M(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    YouTube8M::YouTube8M(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void YouTube8M::load_data()
    {

    }

    void YouTube8M::check_resources()
    {

    }
}
