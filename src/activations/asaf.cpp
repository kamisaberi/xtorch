#include "include/activations/asaf.h"

namespace xt::activations {
    torch::Tensor asaf(torch::Tensor x) {
    }

    torch::Tensor ASAF::forward(torch::Tensor x) const {
        return xt::activations::aglu(x);
    }

}
