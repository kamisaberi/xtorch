#include "include/datasets/time_series/time_series_classification/ucr_time_series_archive.h"

namespace xt::data::datasets
{
    // ---------------------- UCRTimeSeriesArchive ---------------------- //

    UCRTimeSeriesArchive::UCRTimeSeriesArchive(const std::string& root): UCRTimeSeriesArchive::UCRTimeSeriesArchive(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    UCRTimeSeriesArchive::UCRTimeSeriesArchive(const std::string& root, xt::datasets::DataMode mode): UCRTimeSeriesArchive::UCRTimeSeriesArchive(
        root, mode, false, nullptr, nullptr)
    {
    }

    UCRTimeSeriesArchive::UCRTimeSeriesArchive(const std::string& root, xt::datasets::DataMode mode, bool download) :
        UCRTimeSeriesArchive::UCRTimeSeriesArchive(
            root, mode, download, nullptr, nullptr)
    {
    }

    UCRTimeSeriesArchive::UCRTimeSeriesArchive(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : UCRTimeSeriesArchive::UCRTimeSeriesArchive(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    UCRTimeSeriesArchive::UCRTimeSeriesArchive(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void UCRTimeSeriesArchive::load_data()
    {

    }

    void UCRTimeSeriesArchive::check_resources()
    {

    }
}
