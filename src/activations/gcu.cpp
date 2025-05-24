#include "include/activations/gcu.h"

namespace xt::activations {
    torch::Tensor gcu(torch::Tensor x) {
    }

    torch::Tensor GCU::forward(torch::Tensor x) const {
        return xt::activations::gcu(x);
    }

}
