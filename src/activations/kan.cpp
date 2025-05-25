#include "include/activations/kan.h"

namespace xt::activations {
    torch::Tensor kan(torch::Tensor x) {
    }

    torch::Tensor KAN::forward(torch::Tensor x) const {
        return xt::activations::kan(x);
    }

}
