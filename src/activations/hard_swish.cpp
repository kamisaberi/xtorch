#include "include/activations/hard_swish.h"

namespace xt::activations {
    torch::Tensor hard_swich(torch::Tensor x) {
    }

    torch::Tensor HardSwish::forward(torch::Tensor x) const {
        return xt::activations::hard_swich(x);
    }

}
