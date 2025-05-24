#include "include/datasets/audio_processing/query_by_example/quesst14.h"

namespace xt::data::datasets
{
    // ---------------------- QUESST14 ---------------------- //

    QUESST14::QUESST14(const std::string& root): QUESST14::QUESST14(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    QUESST14::QUESST14(const std::string& root, xt::datasets::DataMode mode): QUESST14::QUESST14(
        root, mode, false, nullptr, nullptr)
    {
    }

    QUESST14::QUESST14(const std::string& root, xt::datasets::DataMode mode, bool download) :
        QUESST14::QUESST14(
            root, mode, download, nullptr, nullptr)
    {
    }

    QUESST14::QUESST14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : QUESST14::QUESST14(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    QUESST14::QUESST14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void QUESST14::load_data()
    {

    }

    void QUESST14::check_resources()
    {

    }
}
