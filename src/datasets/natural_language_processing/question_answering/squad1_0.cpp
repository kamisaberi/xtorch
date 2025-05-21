#include "datasets/natural_language_processing/question_answering/squad1_0.h"

namespace xt::data::datasets
{
    // ---------------------- SQuAD10 ---------------------- //

    SQuAD10::SQuAD10(const std::string& root): SQuAD10::SQuAD10(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SQuAD10::SQuAD10(const std::string& root, xt::datasets::DataMode mode): SQuAD10::SQuAD10(
        root, mode, false, nullptr, nullptr)
    {
    }

    SQuAD10::SQuAD10(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SQuAD10::SQuAD10(
            root, mode, download, nullptr, nullptr)
    {
    }

    SQuAD10::SQuAD10(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SQuAD10::SQuAD10(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SQuAD10::SQuAD10(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SQuAD10::load_data()
    {

    }

    void SQuAD10::check_resources()
    {

    }
}
