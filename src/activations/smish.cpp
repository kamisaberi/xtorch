#include "include/activations/smish.h"

namespace xt::activations {
    torch::Tensor smish(torch::Tensor x) {
    }

    torch::Tensor Smish::forward(torch::Tensor x) const {
        return xt::activations::smish(x);
    }

}
