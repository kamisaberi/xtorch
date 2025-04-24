#include "../../../include/datasets/image_classification/fake_data.h"

namespace xt::data::datasets {
    FakeData::FakeData(): FakeData::FakeData(1000, {3, 24, 24}) {
    }

    FakeData::FakeData(size_t size): FakeData::FakeData(size, {3, 24, 24}) {
    }

    FakeData::FakeData(size_t size, vector<int64_t> shape) : BaseDataset(
        "", DataMode::TRAIN, false) {
        throw std::runtime_error("FakeData: FakeData not implemented");
    }


    FakeData::FakeData(size_t size, vector<int64_t> shape, TransformType transforms) : BaseDataset(
        "", DataMode::TRAIN, false, transforms) {
        throw std::runtime_error("FakeData: FakeData not implemented");
    }
}
