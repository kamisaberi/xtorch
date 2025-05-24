#include "include/datasets/computer_vision/semantic_segmentation/oxfordIII_t_pet.h"


namespace xt::data::datasets
{
    // ---------------------- OxfordIIITPet ---------------------- //

    OxfordIIITPet::OxfordIIITPet(const std::string& root): OxfordIIITPet::OxfordIIITPet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string& root, xt::datasets::DataMode mode): OxfordIIITPet::OxfordIIITPet(
        root, mode, false, nullptr, nullptr)
    {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        OxfordIIITPet::OxfordIIITPet(
            root, mode, download, nullptr, nullptr)
    {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : OxfordIIITPet::OxfordIIITPet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    OxfordIIITPet::OxfordIIITPet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void OxfordIIITPet::load_data()
    {

    }

    void OxfordIIITPet::check_resources()
    {

    }
}
