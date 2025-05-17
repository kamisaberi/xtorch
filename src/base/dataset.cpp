#include "base/dataset.h"

namespace xt::datasets {

    Dataset::Dataset() : Dataset(DataMode::TRAIN, nullptr_t, nullptr_t) {}
    Dataset::Dataset(DataMode mode) : Dataset(mode, nullptr_t, nullptr_t) {}
    Dataset::Dataset(DataMode mode, xt::Module transformer) : Dataset(DataMode::TRAIN, transformer, nullptr_t) {}
    Dataset::Dataset(DataMode mode, xt::Module transformer, xt::Module target_transformer)
            : mode(mode),
              transformer(transformer),
              target_transformer(target_transformer) {}

    // BaseDataset::~BaseDataset() {}

    torch::data::Example<> Dataset::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional <size_t> Dataset::size() const {
        return data.size();
    }
}
