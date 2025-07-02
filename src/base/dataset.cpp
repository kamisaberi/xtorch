#include "include/base/base.h"

namespace xt::datasets {

    torch::data::Example<> Dataset::get(size_t index) {
        return {data[index], torch::tensor(targets[index])};
    }

    torch::optional <size_t> Dataset::size() const {
        return data.size();
    }
}
