#include "include/activations/swish.h"

namespace xt::activations {
    torch::Tensor swish(torch::Tensor x) {
    }

    torch::Tensor Swish::forward(torch::Tensor x) const {
        return xt::activations::swish(x);
    }

}
