#include "include/activations/pau.h"

namespace xt::activations {
    torch::Tensor pau(torch::Tensor x) {
    }

    torch::Tensor PAU::forward(torch::Tensor x) const {
        return xt::activations::pau(x);
    }

}
