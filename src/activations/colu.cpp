#include "include/activations/colu.h"

namespace xt::activations {
    torch::Tensor colu(torch::Tensor x) {
    }

    torch::Tensor CoLU::forward(torch::Tensor x) const {
        return xt::activations::colu(x);
    }

}
