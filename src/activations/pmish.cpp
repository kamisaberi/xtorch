#include "include/activations/pmish.h"

namespace xt::activations {
    torch::Tensor pmish(torch::Tensor x) {
    }

    torch::Tensor PMish::forward(torch::Tensor x) const {
        return xt::activations::pmish(x);
    }

}
