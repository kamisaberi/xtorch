#include "include/datasets/audio_processing/speech_synthesis/vctk_092.h"

namespace xt::data::datasets
{
    // ---------------------- VCTK092 ---------------------- //

    VCTK092::VCTK092(const std::string& root): VCTK092::VCTK092(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VCTK092::VCTK092(const std::string& root, xt::datasets::DataMode mode): VCTK092::VCTK092(
        root, mode, false, nullptr, nullptr)
    {
    }

    VCTK092::VCTK092(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VCTK092::VCTK092(
            root, mode, download, nullptr, nullptr)
    {
    }

    VCTK092::VCTK092(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VCTK092::VCTK092(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VCTK092::VCTK092(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VCTK092::load_data()
    {

    }

    void VCTK092::check_resources()
    {

    }
}
