#include "include/datasets/computer_vision/image_classification/fgvc_aircraft.h"

namespace xt::data::datasets
{
    // ---------------------- FGVCAircraft ---------------------- //

    FGVCAircraft::FGVCAircraft(const std::string& root): FGVCAircraft::FGVCAircraft(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FGVCAircraft::FGVCAircraft(const std::string& root, xt::datasets::DataMode mode): FGVCAircraft::FGVCAircraft(
        root, mode, false, nullptr, nullptr)
    {
    }

    FGVCAircraft::FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FGVCAircraft::FGVCAircraft(
            root, mode, download, nullptr, nullptr)
    {
    }

    FGVCAircraft::FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FGVCAircraft::FGVCAircraft(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FGVCAircraft::FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FGVCAircraft::load_data()
    {

    }

    void FGVCAircraft::check_resources()
    {

    }
}
