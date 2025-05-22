#include "datasets/computer_vision/image_classification/fake_data.h"

namespace xt::data::datasets {
    // ---------------------- FakeData ---------------------- //

    FakeData::FakeData(const std::string& root): FakeData::FakeData(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    FakeData::FakeData(const std::string& root, xt::datasets::DataMode mode): FakeData::FakeData(
        root, mode, false, nullptr, nullptr)
    {
    }

    FakeData::FakeData(const std::string& root, xt::datasets::DataMode mode, bool download) :
        FakeData::FakeData(
            root, mode, download, nullptr, nullptr)
    {
    }

    FakeData::FakeData(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : FakeData::FakeData(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    FakeData::FakeData(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void FakeData::load_data()
    {

    }

    void FakeData::check_resources()
    {

    }

}
