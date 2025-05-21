#include "datasets/computer_vision/visual_relationship_detection/visual_genome.h"

namespace xt::data::datasets
{
    // ---------------------- VisualGenome ---------------------- //

    VisualGenome::VisualGenome(const std::string& root): VisualGenome::VisualGenome(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    VisualGenome::VisualGenome(const std::string& root, xt::datasets::DataMode mode): VisualGenome::VisualGenome(
        root, mode, false, nullptr, nullptr)
    {
    }

    VisualGenome::VisualGenome(const std::string& root, xt::datasets::DataMode mode, bool download) :
        VisualGenome::VisualGenome(
            root, mode, download, nullptr, nullptr)
    {
    }

    VisualGenome::VisualGenome(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : VisualGenome::VisualGenome(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    VisualGenome::VisualGenome(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void VisualGenome::load_data()
    {

    }

    void VisualGenome::check_resources()
    {

    }
}
