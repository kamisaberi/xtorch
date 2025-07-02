#include "include/datasets/knowledge_graphs/knowledge_graph_reasoning/freebase.h"

namespace xt::datasets
{
    // ---------------------- FreeBase ---------------------- //

    FreeBase::FreeBase(const std::string& root): FreeBase::FreeBase(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FreeBase::FreeBase(const std::string& root, xt::datasets::DataMode mode): FreeBase::FreeBase(
        root, mode, false, nullptr, nullptr)
    {
    }

    FreeBase::FreeBase(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FreeBase::FreeBase(
            root, mode, download, nullptr, nullptr)
    {
    }

    FreeBase::FreeBase(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FreeBase::FreeBase(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FreeBase::FreeBase(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FreeBase::load_data()
    {

    }

    void FreeBase::check_resources()
    {

    }
}
