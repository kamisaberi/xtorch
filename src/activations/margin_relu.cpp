#include "include/activations/margin_relu.h"

namespace xt::activations {
    torch::Tensor margin_relu(torch::Tensor x) {
    }

    torch::Tensor MarginReLU::forward(torch::Tensor x) const {
        return xt::activations::margin_relu(x);
    }

}
