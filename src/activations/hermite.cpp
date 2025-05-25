#include "include/activations/hermite.h"

namespace xt::activations {
    torch::Tensor hermite(torch::Tensor x) {
    }

    torch::Tensor Hermite::forward(torch::Tensor x) const {
        return xt::activations::hermite(x);
    }

}
