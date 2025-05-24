#include "include/datasets/audio_processing/music_source_separation/mus_db_hq.h"

namespace xt::data::datasets
{
    // ---------------------- MUSDBHQ ---------------------- //

    MUSDBHQ::MUSDBHQ(const std::string& root): MUSDBHQ::MUSDBHQ(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MUSDBHQ::MUSDBHQ(const std::string& root, xt::datasets::DataMode mode): MUSDBHQ::MUSDBHQ(
        root, mode, false, nullptr, nullptr)
    {
    }

    MUSDBHQ::MUSDBHQ(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MUSDBHQ::MUSDBHQ(
            root, mode, download, nullptr, nullptr)
    {
    }

    MUSDBHQ::MUSDBHQ(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MUSDBHQ::MUSDBHQ(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MUSDBHQ::MUSDBHQ(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MUSDBHQ::load_data()
    {

    }

    void MUSDBHQ::check_resources()
    {

    }
}
