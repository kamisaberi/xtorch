#include "datasets/computer_vision/image_text_pretraining/laion_5b.h"

namespace xt::data::datasets
{
    // ---------------------- LAION5B ---------------------- //

    LAION5B::LAION5B(const std::string& root): LAION5B::LAION5B(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LAION5B::LAION5B(const std::string& root, xt::datasets::DataMode mode): LAION5B::LAION5B(
        root, mode, false, nullptr, nullptr)
    {
    }

    LAION5B::LAION5B(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LAION5B::LAION5B(
            root, mode, download, nullptr, nullptr)
    {
    }

    LAION5B::LAION5B(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LAION5B::LAION5B(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LAION5B::LAION5B(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LAION5B::load_data()
    {

    }

    void LAION5B::check_resources()
    {

    }
}
