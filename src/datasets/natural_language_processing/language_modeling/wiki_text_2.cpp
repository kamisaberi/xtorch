#include "datasets/natural_language_processing/language_modeling/wiki_text_2.h"

namespace xt::data::datasets
{
    // ---------------------- WikiTextV2 ---------------------- //

    WikiTextV2::WikiTextV2(const std::string& root): WikiTextV2::WikiTextV2(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WikiTextV2::WikiTextV2(const std::string& root, xt::datasets::DataMode mode): WikiTextV2::WikiTextV2(
        root, mode, false, nullptr, nullptr)
    {
    }

    WikiTextV2::WikiTextV2(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WikiTextV2::WikiTextV2(
            root, mode, download, nullptr, nullptr)
    {
    }

    WikiTextV2::WikiTextV2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WikiTextV2::WikiTextV2(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WikiTextV2::WikiTextV2(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WikiTextV2::load_data()
    {

    }

    void WikiTextV2::check_resources()
    {

    }
}

