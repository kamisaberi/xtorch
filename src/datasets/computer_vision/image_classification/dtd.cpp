#include "include/datasets/computer_vision/image_classification/dtd.h"

namespace xt::datasets
{
    // ---------------------- DTD ---------------------- //

    DTD::DTD(const std::string& root): DTD::DTD(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    DTD::DTD(const std::string& root, xt::datasets::DataMode mode): DTD::DTD(
        root, mode, false, nullptr, nullptr)
    {
    }

    DTD::DTD(const std::string& root, xt::datasets::DataMode mode, bool download) :
        DTD::DTD(
            root, mode, download, nullptr, nullptr)
    {
    }

    DTD::DTD(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : DTD::DTD(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    DTD::DTD(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void DTD::load_data()
    {

    }

    void DTD::check_resources()
    {

    }
}
