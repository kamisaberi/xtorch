#include "datasets/natural_language_processing/sequence_tagging/co_nll2000_chunking.h"

namespace xt::data::datasets
{
    // ---------------------- CoNLL2000Chunking ---------------------- //

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string& root): CoNLL2000Chunking::CoNLL2000Chunking(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string& root, xt::datasets::DataMode mode): CoNLL2000Chunking::CoNLL2000Chunking(
        root, mode, false, nullptr, nullptr)
    {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CoNLL2000Chunking::CoNLL2000Chunking(
            root, mode, download, nullptr, nullptr)
    {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CoNLL2000Chunking::CoNLL2000Chunking(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CoNLL2000Chunking::load_data()
    {

    }

    void CoNLL2000Chunking::check_resources()
    {

    }
}
