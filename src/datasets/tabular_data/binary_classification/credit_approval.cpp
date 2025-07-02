#include "include/datasets/tabular_data/binary_classification/credit_approval.h"

namespace xt::datasets
{
    // ---------------------- CreditApproval ---------------------- //

    CreditApproval::CreditApproval(const std::string& root): CreditApproval::CreditApproval(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CreditApproval::CreditApproval(const std::string& root, xt::datasets::DataMode mode): CreditApproval::CreditApproval(
        root, mode, false, nullptr, nullptr)
    {
    }

    CreditApproval::CreditApproval(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CreditApproval::CreditApproval(
            root, mode, download, nullptr, nullptr)
    {
    }

    CreditApproval::CreditApproval(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CreditApproval::CreditApproval(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CreditApproval::CreditApproval(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CreditApproval::load_data()
    {

    }

    void CreditApproval::check_resources()
    {

    }
}
