#include "datasets/computer_vision/image_classification/caltech101.h"

namespace xt::data::datasets
{
    // ---------------------- Caltech101 ---------------------- //

    Caltech101::Caltech101(const std::string& root): Caltech101::Caltech101(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Caltech101::Caltech101(const std::string& root, xt::datasets::DataMode mode): Caltech101::Caltech101(
        root, mode, false, nullptr, nullptr)
    {
    }

    Caltech101::Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Caltech101::Caltech101(
            root, mode, download, nullptr, nullptr)
    {
    }

    Caltech101::Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Caltech101::Caltech101(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Caltech101::Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Caltech101::load_data()
    {

    }

    void Caltech101::check_resources()
    {

    }
}
