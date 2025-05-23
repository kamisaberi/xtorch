#include "datasets/natural_language_processing/text_classification/yahoo_answers.h"


namespace xt::data::datasets
{
    // ---------------------- YahooAnswers ---------------------- //

    YahooAnswers::YahooAnswers(const std::string& root): YahooAnswers::YahooAnswers(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    YahooAnswers::YahooAnswers(const std::string& root, xt::datasets::DataMode mode): YahooAnswers::YahooAnswers(
        root, mode, false, nullptr, nullptr)
    {
    }

    YahooAnswers::YahooAnswers(const std::string& root, xt::datasets::DataMode mode, bool download) :
        YahooAnswers::YahooAnswers(
            root, mode, download, nullptr, nullptr)
    {
    }

    YahooAnswers::YahooAnswers(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : YahooAnswers::YahooAnswers(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    YahooAnswers::YahooAnswers(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void YahooAnswers::load_data()
    {

    }

    void YahooAnswers::check_resources()
    {

    }
}
