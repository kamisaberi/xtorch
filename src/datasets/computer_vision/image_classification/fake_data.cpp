#include "include/datasets/computer_vision/image_classification/fake_data.h"

namespace xt::datasets
{
    // ---------------------- FakeData ---------------------- //

    FakeData::FakeData(size_t size): FakeData::FakeData(
        size, xt::datasets::DataMode::TRAIN, {3, 24, 24}, nullptr, nullptr)
    {
    }

    FakeData::FakeData(size_t size, xt::datasets::DataMode mode): FakeData::FakeData(
        size, mode, {3, 24, 24}, nullptr, nullptr)
    {
    }

    FakeData::FakeData(size_t size, xt::datasets::DataMode mode, vector<int64_t> shape) : FakeData::FakeData(
        size, mode, shape, nullptr, nullptr)
    {
    }


    FakeData::FakeData(size_t size, xt::datasets::DataMode mode, vector<int64_t> shape,
                       std::unique_ptr<xt::Module> transformer) : FakeData::FakeData(
        size, mode, shape, std::move(transformer), nullptr)
    {
    }

    FakeData::FakeData(size_t size, xt::datasets::DataMode mode, vector<int64_t> shape,
                       std::unique_ptr<xt::Module> transformer,
                       std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        generate_data();
    }



    void FakeData::generate_data()
    {
    }
}
