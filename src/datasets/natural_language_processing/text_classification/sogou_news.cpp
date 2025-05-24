#include "include/datasets/natural_language_processing/text_classification/sogou_news.h"


namespace xt::datasets
{
    // ---------------------- SogouNews ---------------------- //

    SogouNews::SogouNews(const std::string& root): SogouNews::SogouNews(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SogouNews::SogouNews(const std::string& root, xt::datasets::DataMode mode): SogouNews::SogouNews(
        root, mode, false, nullptr, nullptr)
    {
    }

    SogouNews::SogouNews(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SogouNews::SogouNews(
            root, mode, download, nullptr, nullptr)
    {
    }

    SogouNews::SogouNews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SogouNews::SogouNews(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SogouNews::SogouNews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SogouNews::load_data()
    {

    }

    void SogouNews::check_resources()
    {

    }
}
