#include "base/dataset.h"

namespace xt::datasets {

    torch::data::Example<> Dataset::get(size_t index) {
        return {data[index], torch::tensor(labels[index])};
    }

    // Override `size` method to return the number of samples
    torch::optional <size_t> Dataset::size() const {
        return data.size();
    }
}
