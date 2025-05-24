#include "include/datasets/natural_language_processing/question_answering/squad2_0.h"

namespace xt::datasets
{
    // ---------------------- SQuAD20 ---------------------- //

    SQuAD20::SQuAD20(const std::string& root): SQuAD20::SQuAD20(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SQuAD20::SQuAD20(const std::string& root, xt::datasets::DataMode mode): SQuAD20::SQuAD20(
        root, mode, false, nullptr, nullptr)
    {
    }

    SQuAD20::SQuAD20(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SQuAD20::SQuAD20(
            root, mode, download, nullptr, nullptr)
    {
    }

    SQuAD20::SQuAD20(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SQuAD20::SQuAD20(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SQuAD20::SQuAD20(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SQuAD20::load_data()
    {

    }

    void SQuAD20::check_resources()
    {

    }
}
