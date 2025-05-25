#include "include/activations/helu.h"

namespace xt::activations {
    torch::Tensor helu(torch::Tensor x) {
    }

    torch::Tensor HeLU::forward(torch::Tensor x) const {
        return xt::activations::helu(x);
    }

}
