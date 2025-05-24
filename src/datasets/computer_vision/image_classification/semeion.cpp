#include "include/datasets/computer_vision/image_classification/semeion.h"

namespace xt::data::datasets
{
    // ---------------------- SEMEION ---------------------- //

    SEMEION::SEMEION(const std::string& root): SEMEION::SEMEION(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SEMEION::SEMEION(const std::string& root, xt::datasets::DataMode mode): SEMEION::SEMEION(
        root, mode, false, nullptr, nullptr)
    {
    }

    SEMEION::SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SEMEION::SEMEION(
            root, mode, download, nullptr, nullptr)
    {
    }

    SEMEION::SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SEMEION::SEMEION(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SEMEION::SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SEMEION::load_data()
    {

    }

    void SEMEION::check_resources()
    {

    }
}
