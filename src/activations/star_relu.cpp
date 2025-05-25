#include "include/activations/star_relu.h"

namespace xt::activations {
    torch::Tensor star_relu(torch::Tensor x) {
    }

    torch::Tensor StarReLU::forward(torch::Tensor x) const {
        return xt::activations::star_relu(x);
    }

}
