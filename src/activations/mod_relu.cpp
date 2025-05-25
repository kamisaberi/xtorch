#include "include/activations/mod_relu.h"

namespace xt::activations {
    torch::Tensor mod_relu(torch::Tensor x) {
    }

    torch::Tensor ModReLU::forward(torch::Tensor x) const {
        return xt::activations::mod_relu(x);
    }

}
