#include "include/activations/mish.h"

namespace xt::activations {
    torch::Tensor mish(torch::Tensor x) {
    }

    torch::Tensor Mish::forward(torch::Tensor x) const {
        return xt::activations::mish(x);
    }

}
