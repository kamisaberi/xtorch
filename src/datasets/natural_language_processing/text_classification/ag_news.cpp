#include "include/datasets/natural_language_processing/text_classification/ag_news.h"


namespace xt::data::datasets
{
    // ---------------------- AgNews ---------------------- //

    AgNews::AgNews(const std::string& root): AgNews::AgNews(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AgNews::AgNews(const std::string& root, xt::datasets::DataMode mode): AgNews::AgNews(
        root, mode, false, nullptr, nullptr)
    {
    }

    AgNews::AgNews(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AgNews::AgNews(
            root, mode, download, nullptr, nullptr)
    {
    }

    AgNews::AgNews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AgNews::AgNews(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AgNews::AgNews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AgNews::load_data()
    {

    }

    void AgNews::check_resources()
    {

    }
}
