#include "include/activations/poly.h"

namespace xt::activations {
    torch::Tensor poly(torch::Tensor x) {
    }

    torch::Tensor Poly::forward(torch::Tensor x) const {
        return xt::activations::poly(x);
    }

}
