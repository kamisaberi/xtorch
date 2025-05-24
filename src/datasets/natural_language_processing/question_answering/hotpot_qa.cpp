#include "include/datasets/natural_language_processing/question_answering/hotpot_qa.h"

namespace xt::data::datasets
{
    // ---------------------- HotpotQA ---------------------- //

    HotpotQA::HotpotQA(const std::string& root): HotpotQA::HotpotQA(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    HotpotQA::HotpotQA(const std::string& root, xt::datasets::DataMode mode): HotpotQA::HotpotQA(
        root, mode, false, nullptr, nullptr)
    {
    }

    HotpotQA::HotpotQA(const std::string& root, xt::datasets::DataMode mode, bool download) :
        HotpotQA::HotpotQA(
            root, mode, download, nullptr, nullptr)
    {
    }

    HotpotQA::HotpotQA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : HotpotQA::HotpotQA(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    HotpotQA::HotpotQA(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void HotpotQA::load_data()
    {

    }

    void HotpotQA::check_resources()
    {

    }
}
