#include "include/datasets/natural_language_processing/math_word_problems/gsm8k.h"

namespace xt::datasets
{
    // ---------------------- GSM8K ---------------------- //

    GSM8K::GSM8K(const std::string& root): GSM8K::GSM8K(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    GSM8K::GSM8K(const std::string& root, xt::datasets::DataMode mode): GSM8K::GSM8K(
        root, mode, false, nullptr, nullptr)
    {
    }

    GSM8K::GSM8K(const std::string& root, xt::datasets::DataMode mode, bool download) :
        GSM8K::GSM8K(
            root, mode, download, nullptr, nullptr)
    {
    }

    GSM8K::GSM8K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : GSM8K::GSM8K(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    GSM8K::GSM8K(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void GSM8K::load_data()
    {

    }

    void GSM8K::check_resources()
    {

    }
}
