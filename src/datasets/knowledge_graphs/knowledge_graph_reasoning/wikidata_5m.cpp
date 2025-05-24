#include "include/datasets/knowledge_graphs/knowledge_graph_reasoning/wikidata_5m.h"

namespace xt::datasets
{
    // ---------------------- Wikidata5M ---------------------- //

    Wikidata5M::Wikidata5M(const std::string& root): Wikidata5M::Wikidata5M(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Wikidata5M::Wikidata5M(const std::string& root, xt::datasets::DataMode mode): Wikidata5M::Wikidata5M(
        root, mode, false, nullptr, nullptr)
    {
    }

    Wikidata5M::Wikidata5M(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Wikidata5M::Wikidata5M(
            root, mode, download, nullptr, nullptr)
    {
    }

    Wikidata5M::Wikidata5M(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Wikidata5M::Wikidata5M(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Wikidata5M::Wikidata5M(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Wikidata5M::load_data()
    {

    }

    void Wikidata5M::check_resources()
    {

    }
}
