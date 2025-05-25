#include "include/activations/gumbel.h"

namespace xt::activations {
    torch::Tensor gumbel(torch::Tensor x) {
    }

    torch::Tensor Gumbel::forward(torch::Tensor x) const {
        return xt::activations::gumbel(x);
    }

}
