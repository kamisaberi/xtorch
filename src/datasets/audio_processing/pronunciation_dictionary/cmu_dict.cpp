#include "include/datasets/audio_processing/pronunciation_dictionary/cmu_dict.h"

namespace xt::datasets
{

    CMUDict::CMUDict(const std::string& root): CMUDict::CMUDict(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CMUDict::CMUDict(const std::string& root, xt::datasets::DataMode mode): CMUDict::CMUDict(
        root, mode, false, nullptr, nullptr)
    {
    }

    CMUDict::CMUDict(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CMUDict::CMUDict(
            root, mode, download, nullptr, nullptr)
    {
    }

    CMUDict::CMUDict(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CMUDict::CMUDict(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CMUDict::CMUDict(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CMUDict::load_data()
    {

    }

    void CMUDict::check_resources()
    {

    }
}
