#include "include/activations/norm_lin_comb.h"

namespace xt::activations {
    torch::Tensor norm_lin_comb(torch::Tensor x) {
    }

    torch::Tensor NormLinComb::forward(torch::Tensor x) const {
        return xt::activations::norm_lin_comb(x);
    }

}
