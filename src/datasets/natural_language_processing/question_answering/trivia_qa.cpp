#include "include/datasets/natural_language_processing/question_answering/trivia_qa.h"

namespace xt::data::datasets
{
    // ---------------------- TriviaQA ---------------------- //

    TriviaQA::TriviaQA(const std::string& root): TriviaQA::TriviaQA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    TriviaQA::TriviaQA(const std::string& root, xt::datasets::DataMode mode): TriviaQA::TriviaQA(
        root, mode, false, nullptr, nullptr)
    {
    }

    TriviaQA::TriviaQA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        TriviaQA::TriviaQA(
            root, mode, download, nullptr, nullptr)
    {
    }

    TriviaQA::TriviaQA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : TriviaQA::TriviaQA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    TriviaQA::TriviaQA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void TriviaQA::load_data()
    {

    }

    void TriviaQA::check_resources()
    {

    }
}
