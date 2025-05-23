#include "datasets/computer_vision/image_pair_tasks/photo_tour.h"

namespace xt::data::datasets
{
    // ---------------------- PhotoTour ---------------------- //

    PhotoTour::PhotoTour(const std::string& root): PhotoTour::PhotoTour(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    PhotoTour::PhotoTour(const std::string& root, xt::datasets::DataMode mode): PhotoTour::PhotoTour(
        root, mode, false, nullptr, nullptr)
    {
    }

    PhotoTour::PhotoTour(const std::string& root, xt::datasets::DataMode mode, bool download) :
        PhotoTour::PhotoTour(
            root, mode, download, nullptr, nullptr)
    {
    }

    PhotoTour::PhotoTour(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : PhotoTour::PhotoTour(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    PhotoTour::PhotoTour(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void PhotoTour::load_data()
    {

    }

    void PhotoTour::check_resources()
    {

    }
}
