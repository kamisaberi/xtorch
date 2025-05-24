#include "include/datasets/natural_language_processing/text_summarization/xsum.h"

namespace xt::data::datasets
{
    // ---------------------- XSum ---------------------- //

    XSum::XSum(const std::string& root): XSum::XSum(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    XSum::XSum(const std::string& root, xt::datasets::DataMode mode): XSum::XSum(
        root, mode, false, nullptr, nullptr)
    {
    }

    XSum::XSum(const std::string& root, xt::datasets::DataMode mode, bool download) :
        XSum::XSum(
            root, mode, download, nullptr, nullptr)
    {
    }

    XSum::XSum(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : XSum::XSum(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    XSum::XSum(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void XSum::load_data()
    {

    }

    void XSum::check_resources()
    {

    }
}
