#include "datasets/computer_vision/instance_segmentation/lvis.h"

namespace xt::data::datasets
{
    // ---------------------- LVIS ---------------------- //

    LVIS::LVIS(const std::string& root): LVIS::LVIS(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LVIS::LVIS(const std::string& root, xt::datasets::DataMode mode): LVIS::LVIS(
        root, mode, false, nullptr, nullptr)
    {
    }

    LVIS::LVIS(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LVIS::LVIS(
            root, mode, download, nullptr, nullptr)
    {
    }

    LVIS::LVIS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LVIS::LVIS(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LVIS::LVIS(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LVIS::load_data()
    {

    }

    void LVIS::check_resources()
    {

    }
}
