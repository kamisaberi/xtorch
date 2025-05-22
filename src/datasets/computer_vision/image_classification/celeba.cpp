#include "datasets/computer_vision/image_classification/celeba.h"

namespace xt::data::datasets
{
    // ---------------------- CelebA ---------------------- //

    CelebA::CelebA(const std::string& root): CelebA::CelebA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode): CelebA::CelebA(
        root, mode, false, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CelebA::CelebA(
            root, mode, download, nullptr, nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CelebA::CelebA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CelebA::CelebA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CelebA::load_data()
    {

    }

    void CelebA::check_resources()
    {

    }
}
