#include "include/datasets/tabular_data/binary_classification/breast_cancer_wisconsin.h"

namespace xt::datasets
{
    // ---------------------- BreastCancerWisconsin ---------------------- //

    BreastCancerWisconsin::BreastCancerWisconsin(const std::string& root): BreastCancerWisconsin::BreastCancerWisconsin(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    BreastCancerWisconsin::BreastCancerWisconsin(const std::string& root, xt::datasets::DataMode mode): BreastCancerWisconsin::BreastCancerWisconsin(
        root, mode, false, nullptr, nullptr)
    {
    }

    BreastCancerWisconsin::BreastCancerWisconsin(const std::string& root, xt::datasets::DataMode mode, bool download) :
        BreastCancerWisconsin::BreastCancerWisconsin(
            root, mode, download, nullptr, nullptr)
    {
    }

    BreastCancerWisconsin::BreastCancerWisconsin(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : BreastCancerWisconsin::BreastCancerWisconsin(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    BreastCancerWisconsin::BreastCancerWisconsin(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void BreastCancerWisconsin::load_data()
    {

    }

    void BreastCancerWisconsin::check_resources()
    {

    }
}
