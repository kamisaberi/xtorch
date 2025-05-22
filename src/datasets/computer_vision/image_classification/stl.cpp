#include "datasets/computer_vision/image_classification/stl.h"

namespace xt::data::datasets
{
    // ---------------------- STL10 ---------------------- //

    STL10::STL10(const std::string& root): STL10::STL10(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    STL10::STL10(const std::string& root, xt::datasets::DataMode mode): STL10::STL10(
        root, mode, false, nullptr, nullptr)
    {
    }

    STL10::STL10(const std::string& root, xt::datasets::DataMode mode, bool download) :
        STL10::STL10(
            root, mode, download, nullptr, nullptr)
    {
    }

    STL10::STL10(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : STL10::STL10(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    STL10::STL10(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void STL10::load_data()
    {

    }

    void STL10::check_resources()
    {

    }
}
