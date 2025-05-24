#include "include/datasets/computer_vision/image_generation/ffhq.h"

namespace xt::datasets
{
    // ---------------------- FFHQ ---------------------- //

    FFHQ::FFHQ(const std::string& root): FFHQ::FFHQ(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FFHQ::FFHQ(const std::string& root, xt::datasets::DataMode mode): FFHQ::FFHQ(
        root, mode, false, nullptr, nullptr)
    {
    }

    FFHQ::FFHQ(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FFHQ::FFHQ(
            root, mode, download, nullptr, nullptr)
    {
    }

    FFHQ::FFHQ(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FFHQ::FFHQ(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FFHQ::FFHQ(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FFHQ::load_data()
    {

    }

    void FFHQ::check_resources()
    {

    }
}
