#include "include/datasets/computer_vision/optical_flow_estimation/sintel.h"


namespace xt::data::datasets
{
    // ---------------------- Sintel ---------------------- //

    Sintel::Sintel(const std::string& root): Sintel::Sintel(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Sintel::Sintel(const std::string& root, xt::datasets::DataMode mode): Sintel::Sintel(
        root, mode, false, nullptr, nullptr)
    {
    }

    Sintel::Sintel(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Sintel::Sintel(
            root, mode, download, nullptr, nullptr)
    {
    }

    Sintel::Sintel(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Sintel::Sintel(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Sintel::Sintel(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Sintel::load_data()
    {

    }

    void Sintel::check_resources()
    {

    }
}
