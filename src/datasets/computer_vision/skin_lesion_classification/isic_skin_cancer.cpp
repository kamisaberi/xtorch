#include "datasets/computer_vision/skin_lesion_classification/isic_skin_cancer.h"

namespace xt::data::datasets
{
    // ---------------------- ISICSkinCancer ---------------------- //

    ISICSkinCancer::ISICSkinCancer(const std::string& root): ISICSkinCancer::ISICSkinCancer(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ISICSkinCancer::ISICSkinCancer(const std::string& root, xt::datasets::DataMode mode): ISICSkinCancer::ISICSkinCancer(
        root, mode, false, nullptr, nullptr)
    {
    }

    ISICSkinCancer::ISICSkinCancer(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ISICSkinCancer::ISICSkinCancer(
            root, mode, download, nullptr, nullptr)
    {
    }

    ISICSkinCancer::ISICSkinCancer(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ISICSkinCancer::ISICSkinCancer(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ISICSkinCancer::ISICSkinCancer(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ISICSkinCancer::load_data()
    {

    }

    void ISICSkinCancer::check_resources()
    {

    }
}
