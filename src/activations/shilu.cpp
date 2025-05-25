#include "include/activations/shilu.h"

namespace xt::activations {
    torch::Tensor shilu(torch::Tensor x) {
    }

    torch::Tensor ShiLU::forward(torch::Tensor x) const {
        return xt::activations::shilu(x);
    }

}
