#include "datasets/natural_language_processing/text_summarization/cnn_daily_mail.h"

namespace xt::data::datasets
{
    // ---------------------- CNNDailyMail ---------------------- //

    CNNDailyMail::CNNDailyMail(const std::string& root): CNNDailyMail::CNNDailyMail(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CNNDailyMail::CNNDailyMail(const std::string& root, xt::datasets::DataMode mode): CNNDailyMail::CNNDailyMail(
        root, mode, false, nullptr, nullptr)
    {
    }

    CNNDailyMail::CNNDailyMail(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CNNDailyMail::CNNDailyMail(
            root, mode, download, nullptr, nullptr)
    {
    }

    CNNDailyMail::CNNDailyMail(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CNNDailyMail::CNNDailyMail(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CNNDailyMail::CNNDailyMail(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CNNDailyMail::load_data()
    {

    }

    void CNNDailyMail::check_resources()
    {

    }
}
