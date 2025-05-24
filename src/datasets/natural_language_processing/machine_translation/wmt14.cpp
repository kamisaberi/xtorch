#include "include/datasets/natural_language_processing/machine_translation/wmt14.h"


namespace xt::data::datasets
{
    // ---------------------- WMT14 ---------------------- //

    WMT14::WMT14(const std::string& root): WMT14::WMT14(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    WMT14::WMT14(const std::string& root, xt::datasets::DataMode mode): WMT14::WMT14(
        root, mode, false, nullptr, nullptr)
    {
    }

    WMT14::WMT14(const std::string& root, xt::datasets::DataMode mode, bool download) :
        WMT14::WMT14(
            root, mode, download, nullptr, nullptr)
    {
    }

    WMT14::WMT14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : WMT14::WMT14(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    WMT14::WMT14(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void WMT14::load_data()
    {

    }

    void WMT14::check_resources()
    {

    }
}
