#include "datasets/computer_vision/image_classification/omniglot.h"

namespace xt::data::datasets
{
    // ---------------------- Omniglot ---------------------- //

    Omniglot::Omniglot(const std::string& root): Omniglot::Omniglot(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Omniglot::Omniglot(const std::string& root, xt::datasets::DataMode mode): Omniglot::Omniglot(
        root, mode, false, nullptr, nullptr)
    {
    }

    Omniglot::Omniglot(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Omniglot::Omniglot(
            root, mode, download, nullptr, nullptr)
    {
    }

    Omniglot::Omniglot(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Omniglot::Omniglot(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Omniglot::Omniglot(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Omniglot::load_data()
    {

    }

    void Omniglot::check_resources()
    {

    }
}
