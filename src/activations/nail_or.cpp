#include "include/activations/nail_or.h"

namespace xt::activations {
    torch::Tensor nail_or(torch::Tensor x) {
    }

    torch::Tensor NailOr::forward(torch::Tensor x) const {
        return xt::activations::nail_or(x);
    }

}
