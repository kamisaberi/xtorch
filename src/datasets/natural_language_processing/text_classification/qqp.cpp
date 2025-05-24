#include "include/datasets/natural_language_processing/text_classification/qqp.h"


namespace xt::data::datasets
{
    // ---------------------- QQP ---------------------- //

    QQP::QQP(const std::string& root): QQP::QQP(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    QQP::QQP(const std::string& root, xt::datasets::DataMode mode): QQP::QQP(
        root, mode, false, nullptr, nullptr)
    {
    }

    QQP::QQP(const std::string& root, xt::datasets::DataMode mode, bool download) :
        QQP::QQP(
            root, mode, download, nullptr, nullptr)
    {
    }

    QQP::QQP(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : QQP::QQP(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    QQP::QQP(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void QQP::load_data()
    {

    }

    void QQP::check_resources()
    {

    }
}
