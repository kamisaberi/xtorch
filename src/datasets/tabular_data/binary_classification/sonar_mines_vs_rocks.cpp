#include "include/datasets/tabular_data/binary_classification/sonar_mines_vs_rocks.h"

namespace xt::data::datasets
{
    // ---------------------- SonarMinesVsRocks ---------------------- //

    SonarMinesVsRocks::SonarMinesVsRocks(const std::string& root): SonarMinesVsRocks::SonarMinesVsRocks(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SonarMinesVsRocks::SonarMinesVsRocks(const std::string& root, xt::datasets::DataMode mode): SonarMinesVsRocks::SonarMinesVsRocks(
        root, mode, false, nullptr, nullptr)
    {
    }

    SonarMinesVsRocks::SonarMinesVsRocks(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SonarMinesVsRocks::SonarMinesVsRocks(
            root, mode, download, nullptr, nullptr)
    {
    }

    SonarMinesVsRocks::SonarMinesVsRocks(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SonarMinesVsRocks::SonarMinesVsRocks(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SonarMinesVsRocks::SonarMinesVsRocks(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SonarMinesVsRocks::load_data()
    {

    }

    void SonarMinesVsRocks::check_resources()
    {

    }
}
