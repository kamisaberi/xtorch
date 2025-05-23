#include "datasets/audio_processing/speech_synthesis/cmu_arctic.h"


namespace xt::data::datasets
{
    // ---------------------- CMUArctic ---------------------- //

    CMUArctic::CMUArctic(const std::string& root): CMUArctic::CMUArctic(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CMUArctic::CMUArctic(const std::string& root, xt::datasets::DataMode mode): CMUArctic::CMUArctic(
        root, mode, false, nullptr, nullptr)
    {
    }

    CMUArctic::CMUArctic(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CMUArctic::CMUArctic(
            root, mode, download, nullptr, nullptr)
    {
    }

    CMUArctic::CMUArctic(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CMUArctic::CMUArctic(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CMUArctic::CMUArctic(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CMUArctic::load_data()
    {

    }

    void CMUArctic::check_resources()
    {

    }
}
