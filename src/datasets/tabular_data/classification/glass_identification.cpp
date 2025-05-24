#include "include/datasets/tabular_data/classification/glass_identification.h"

namespace xt::datasets
{
    // ---------------------- GlassIdentification ---------------------- //

    GlassIdentification::GlassIdentification(const std::string& root): GlassIdentification::GlassIdentification(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    GlassIdentification::GlassIdentification(const std::string& root, xt::datasets::DataMode mode): GlassIdentification::GlassIdentification(
        root, mode, false, nullptr, nullptr)
    {
    }

    GlassIdentification::GlassIdentification(const std::string& root, xt::datasets::DataMode mode, bool download) :
        GlassIdentification::GlassIdentification(
            root, mode, download, nullptr, nullptr)
    {
    }

    GlassIdentification::GlassIdentification(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : GlassIdentification::GlassIdentification(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    GlassIdentification::GlassIdentification(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void GlassIdentification::load_data()
    {

    }

    void GlassIdentification::check_resources()
    {

    }
}
