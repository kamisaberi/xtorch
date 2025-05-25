#include "include/activations/taaf.h"

namespace xt::activations {
    torch::Tensor taaf(torch::Tensor x) {
    }

    torch::Tensor TAAF::forward(torch::Tensor x) const {
        return xt::activations::taaf(x);
    }

}
