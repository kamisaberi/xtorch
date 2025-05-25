#include "include/activations/phish.h"

namespace xt::activations {
    torch::Tensor phish(torch::Tensor x) {
    }

    torch::Tensor Phish::forward(torch::Tensor x) const {
        return xt::activations::phish(x);
    }

}
