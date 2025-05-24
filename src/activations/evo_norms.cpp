#include "include/activations/evo_norms.h"

namespace xt::activations {
    torch::Tensor evo_norms(torch::Tensor x) {
    }

    torch::Tensor EvoNorms::forward(torch::Tensor x) const {
        return xt::activations::evo_norms(x);
    }

}
