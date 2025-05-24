#include "include/datasets/natural_language_processing/question_answering/natural_questions.h"

namespace xt::datasets
{
    // ---------------------- NaturalQuestions ---------------------- //

    NaturalQuestions::NaturalQuestions(const std::string& root): NaturalQuestions::NaturalQuestions(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    NaturalQuestions::NaturalQuestions(const std::string& root, xt::datasets::DataMode mode): NaturalQuestions::NaturalQuestions(
        root, mode, false, nullptr, nullptr)
    {
    }

    NaturalQuestions::NaturalQuestions(const std::string& root, xt::datasets::DataMode mode, bool download) :
        NaturalQuestions::NaturalQuestions(
            root, mode, download, nullptr, nullptr)
    {
    }

    NaturalQuestions::NaturalQuestions(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : NaturalQuestions::NaturalQuestions(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    NaturalQuestions::NaturalQuestions(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void NaturalQuestions::load_data()
    {

    }

    void NaturalQuestions::check_resources()
    {

    }
}
