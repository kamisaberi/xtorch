#include "include/activations/srelu.h"

namespace xt::activations {
    torch::Tensor srelu(torch::Tensor x) {
    }

    torch::Tensor SReLU::forward(torch::Tensor x) const {
        return xt::activations::srelu(x);
    }

}
