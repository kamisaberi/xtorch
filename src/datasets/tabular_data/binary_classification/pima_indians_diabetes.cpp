#include "include/datasets/tabular_data/binary_classification/pima_indians_diabetes.h"

namespace xt::data::datasets
{
    // ---------------------- PimaIndiansDiabetes ---------------------- //

    PimaIndiansDiabetes::PimaIndiansDiabetes(const std::string& root): PimaIndiansDiabetes::PimaIndiansDiabetes(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PimaIndiansDiabetes::PimaIndiansDiabetes(const std::string& root, xt::datasets::DataMode mode): PimaIndiansDiabetes::PimaIndiansDiabetes(
        root, mode, false, nullptr, nullptr)
    {
    }

    PimaIndiansDiabetes::PimaIndiansDiabetes(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PimaIndiansDiabetes::PimaIndiansDiabetes(
            root, mode, download, nullptr, nullptr)
    {
    }

    PimaIndiansDiabetes::PimaIndiansDiabetes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PimaIndiansDiabetes::PimaIndiansDiabetes(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PimaIndiansDiabetes::PimaIndiansDiabetes(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PimaIndiansDiabetes::load_data()
    {

    }

    void PimaIndiansDiabetes::check_resources()
    {

    }
}
