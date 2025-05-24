#include "include/datasets/reinforcement_learning/reinforcement_learning/atari_2600_ale.h"

namespace xt::datasets
{
    // ---------------------- Atari2600ALE ---------------------- //

    Atari2600ALE::Atari2600ALE(const std::string& root): Atari2600ALE::Atari2600ALE(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Atari2600ALE::Atari2600ALE(const std::string& root, xt::datasets::DataMode mode): Atari2600ALE::Atari2600ALE(
        root, mode, false, nullptr, nullptr)
    {
    }

    Atari2600ALE::Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Atari2600ALE::Atari2600ALE(
            root, mode, download, nullptr, nullptr)
    {
    }

    Atari2600ALE::Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Atari2600ALE::Atari2600ALE(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Atari2600ALE::Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Atari2600ALE::load_data()
    {

    }

    void Atari2600ALE::check_resources()
    {

    }
}
