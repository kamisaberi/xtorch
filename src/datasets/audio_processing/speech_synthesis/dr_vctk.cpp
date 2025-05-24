#include "include/datasets/audio_processing/speech_synthesis/dr_vctk.h"


namespace xt::datasets
{
    // ---------------------- VCTK ---------------------- //

    VCTK::VCTK(const std::string& root): VCTK::VCTK(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VCTK::VCTK(const std::string& root, xt::datasets::DataMode mode): VCTK::VCTK(
        root, mode, false, nullptr, nullptr)
    {
    }

    VCTK::VCTK(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VCTK::VCTK(
            root, mode, download, nullptr, nullptr)
    {
    }

    VCTK::VCTK(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VCTK::VCTK(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VCTK::VCTK(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VCTK::load_data()
    {

    }

    void VCTK::check_resources()
    {

    }
}
