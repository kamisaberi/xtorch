#include "include/datasets/natural_language_processing/text_classification/cola.h"

namespace xt::data::datasets
{
    // ---------------------- COLA ---------------------- //

    COLA::COLA(const std::string& root): COLA::COLA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    COLA::COLA(const std::string& root, xt::datasets::DataMode mode): COLA::COLA(
        root, mode, false, nullptr, nullptr)
    {
    }

    COLA::COLA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        COLA::COLA(
            root, mode, download, nullptr, nullptr)
    {
    }

    COLA::COLA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : COLA::COLA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    COLA::COLA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void COLA::load_data()
    {

    }

    void COLA::check_resources()
    {

    }
}
