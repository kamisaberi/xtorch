#include "include/datasets/graph_data/node_classification/cora.h"

namespace xt::data::datasets
{
    // ---------------------- Cora ---------------------- //

    Cora::Cora(const std::string& root): Cora::Cora(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Cora::Cora(const std::string& root, xt::datasets::DataMode mode): Cora::Cora(
        root, mode, false, nullptr, nullptr)
    {
    }

    Cora::Cora(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Cora::Cora(
            root, mode, download, nullptr, nullptr)
    {
    }

    Cora::Cora(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Cora::Cora(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Cora::Cora(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Cora::load_data()
    {

    }

    void Cora::check_resources()
    {

    }
}
