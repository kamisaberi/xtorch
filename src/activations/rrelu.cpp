#include "include/activations/rrelu.h"

namespace xt::activations {
    torch::Tensor rrelu(torch::Tensor x) {
    }

    torch::Tensor RReLU::forward(torch::Tensor x) const {
        return xt::activations::rrelu(x);
    }

}
