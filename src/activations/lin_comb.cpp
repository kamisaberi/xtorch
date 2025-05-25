#include "include/activations/lin_comb.h"

namespace xt::activations {
    torch::Tensor lin_comb(torch::Tensor x) {
    }

    torch::Tensor LinComb::forward(torch::Tensor x) const {
        return xt::activations::lin_comb(x);
    }

}
