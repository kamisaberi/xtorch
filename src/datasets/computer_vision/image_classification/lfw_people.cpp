#include "include/datasets/computer_vision/image_classification/lfw_people.h"

namespace xt::data::datasets
{
    // ---------------------- LFWPeople ---------------------- //

    LFWPeople::LFWPeople(const std::string& root): LFWPeople::LFWPeople(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    LFWPeople::LFWPeople(const std::string& root, xt::datasets::DataMode mode): LFWPeople::LFWPeople(
        root, mode, false, nullptr, nullptr)
    {
    }

    LFWPeople::LFWPeople(const std::string& root, xt::datasets::DataMode mode, bool download) :
        LFWPeople::LFWPeople(
            root, mode, download, nullptr, nullptr)
    {
    }

    LFWPeople::LFWPeople(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : LFWPeople::LFWPeople(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    LFWPeople::LFWPeople(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void LFWPeople::load_data()
    {

    }

    void LFWPeople::check_resources()
    {

    }
}
