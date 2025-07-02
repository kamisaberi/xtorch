#include "include/datasets/tabular_data/classification/car_evaluation.h"

namespace xt::datasets
{
    // ---------------------- CarEvaluation ---------------------- //

    CarEvaluation::CarEvaluation(const std::string& root): CarEvaluation::CarEvaluation(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    CarEvaluation::CarEvaluation(const std::string& root, xt::datasets::DataMode mode): CarEvaluation::CarEvaluation(
        root, mode, false, nullptr, nullptr)
    {
    }

    CarEvaluation::CarEvaluation(const std::string& root, xt::datasets::DataMode mode, bool download) :
        CarEvaluation::CarEvaluation(
            root, mode, download, nullptr, nullptr)
    {
    }

    CarEvaluation::CarEvaluation(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : CarEvaluation::CarEvaluation(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    CarEvaluation::CarEvaluation(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void CarEvaluation::load_data()
    {

    }

    void CarEvaluation::check_resources()
    {

    }
}
