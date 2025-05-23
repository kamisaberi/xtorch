#include "datasets/natural_language_processing/text_classification/db_pedia.h"


namespace xt::data::datasets
{
    // ---------------------- DBPedia ---------------------- //

    DBPedia::DBPedia(const std::string& root): DBPedia::DBPedia(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    DBPedia::DBPedia(const std::string& root, xt::datasets::DataMode mode): DBPedia::DBPedia(
        root, mode, false, nullptr, nullptr)
    {
    }

    DBPedia::DBPedia(const std::string& root, xt::datasets::DataMode mode, bool download) :
        DBPedia::DBPedia(
            root, mode, download, nullptr, nullptr)
    {
    }

    DBPedia::DBPedia(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : DBPedia::DBPedia(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    DBPedia::DBPedia(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void DBPedia::load_data()
    {

    }

    void DBPedia::check_resources()
    {

    }
}
