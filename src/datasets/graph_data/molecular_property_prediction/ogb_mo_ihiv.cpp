#include "include/datasets/graph_data/molecular_property_prediction/ogb_mo_ihiv.h"

namespace xt::data::datasets
{
    // ---------------------- OGBMolHIV ---------------------- //

    OGBMolHIV::OGBMolHIV(const std::string& root): OGBMolHIV::OGBMolHIV(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    OGBMolHIV::OGBMolHIV(const std::string& root, xt::datasets::DataMode mode): OGBMolHIV::OGBMolHIV(
        root, mode, false, nullptr, nullptr)
    {
    }

    OGBMolHIV::OGBMolHIV(const std::string& root, xt::datasets::DataMode mode, bool download) :
        OGBMolHIV::OGBMolHIV(
            root, mode, download, nullptr, nullptr)
    {
    }

    OGBMolHIV::OGBMolHIV(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : OGBMolHIV::OGBMolHIV(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    OGBMolHIV::OGBMolHIV(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void OGBMolHIV::load_data()
    {

    }

    void OGBMolHIV::check_resources()
    {

    }
}
