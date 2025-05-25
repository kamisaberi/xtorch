#include "include/activations/kaf.h"

namespace xt::activations {
    torch::Tensor kaf(torch::Tensor x) {
    }

    torch::Tensor KAF::forward(torch::Tensor x) const {
        return xt::activations::kaf(x);
    }

}
