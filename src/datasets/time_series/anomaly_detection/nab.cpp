#include "datasets/time_series/anomaly_detection/nab.h"

namespace xt::data::datasets
{
    // ---------------------- NAB ---------------------- //

    NAB::NAB(const std::string& root): NAB::NAB(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    NAB::NAB(const std::string& root, xt::datasets::DataMode mode): NAB::NAB(
        root, mode, false, nullptr, nullptr)
    {
    }

    NAB::NAB(const std::string& root, xt::datasets::DataMode mode, bool download) :
        NAB::NAB(
            root, mode, download, nullptr, nullptr)
    {
    }

    NAB::NAB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : NAB::NAB(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    NAB::NAB(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void NAB::load_data()
    {

    }

    void NAB::check_resources()
    {

    }
}
