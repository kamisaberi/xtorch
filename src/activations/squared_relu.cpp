#include "include/activations/squared_relu.h"

namespace xt::activations {
    torch::Tensor squared_relu(torch::Tensor x) {
    }

    torch::Tensor SquaredReLU::forward(torch::Tensor x) const {
        return xt::activations::squared_relu(x);
    }

}
