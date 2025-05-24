#include "include/datasets/natural_language_processing/language_modeling/wiki_text103.h"


namespace xt::datasets
{
    // ---------------------- WikiText103 ---------------------- //

    WikiText103::WikiText103(const std::string& root): WikiText103::WikiText103(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WikiText103::WikiText103(const std::string& root, xt::datasets::DataMode mode): WikiText103::WikiText103(
        root, mode, false, nullptr, nullptr)
    {
    }

    WikiText103::WikiText103(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WikiText103::WikiText103(
            root, mode, download, nullptr, nullptr)
    {
    }

    WikiText103::WikiText103(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WikiText103::WikiText103(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WikiText103::WikiText103(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WikiText103::load_data()
    {

    }

    void WikiText103::check_resources()
    {

    }
}
