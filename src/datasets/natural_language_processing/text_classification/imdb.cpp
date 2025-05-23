#include "datasets/natural_language_processing/text_classification/imdb.h"

namespace xt::data::datasets
{
    // ---------------------- IMDB ---------------------- //

    IMDB::IMDB(const std::string& root): IMDB::IMDB(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    IMDB::IMDB(const std::string& root, xt::datasets::DataMode mode): IMDB::IMDB(
        root, mode, false, nullptr, nullptr)
    {
    }

    IMDB::IMDB(const std::string& root, xt::datasets::DataMode mode, bool download) :
        IMDB::IMDB(
            root, mode, download, nullptr, nullptr)
    {
    }

    IMDB::IMDB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : IMDB::IMDB(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    IMDB::IMDB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void IMDB::load_data()
    {

    }

    void IMDB::check_resources()
    {

    }
}
