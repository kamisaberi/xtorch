#include "include/datasets/natural_language_processing/dialogue_generation/daily_dialog.h"

namespace xt::datasets
{
    // ---------------------- DailyDialog ---------------------- //

    DailyDialog::DailyDialog(const std::string& root): DailyDialog::DailyDialog(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    DailyDialog::DailyDialog(const std::string& root, xt::datasets::DataMode mode): DailyDialog::DailyDialog(
        root, mode, false, nullptr, nullptr)
    {
    }

    DailyDialog::DailyDialog(const std::string& root, xt::datasets::DataMode mode, bool download) :
        DailyDialog::DailyDialog(
            root, mode, download, nullptr, nullptr)
    {
    }

    DailyDialog::DailyDialog(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : DailyDialog::DailyDialog(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    DailyDialog::DailyDialog(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void DailyDialog::load_data()
    {

    }

    void DailyDialog::check_resources()
    {

    }
}
