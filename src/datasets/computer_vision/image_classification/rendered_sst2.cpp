#include "datasets/computer_vision/image_classification/rendered_sst2.h"

namespace xt::data::datasets
{
    // ---------------------- RenderedSST2 ---------------------- //

    RenderedSST2::RenderedSST2(const std::string& root): RenderedSST2::RenderedSST2(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    RenderedSST2::RenderedSST2(const std::string& root, xt::datasets::DataMode mode): RenderedSST2::RenderedSST2(
        root, mode, false, nullptr, nullptr)
    {
    }

    RenderedSST2::RenderedSST2(const std::string& root, xt::datasets::DataMode mode, bool download) :
        RenderedSST2::RenderedSST2(
            root, mode, download, nullptr, nullptr)
    {
    }

    RenderedSST2::RenderedSST2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : RenderedSST2::RenderedSST2(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    RenderedSST2::RenderedSST2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void RenderedSST2::load_data()
    {

    }

    void RenderedSST2::check_resources()
    {

    }
}
