#include "include/activations/aglu.h"

namespace xt::activations {
    torch::Tensor aglu(torch::Tensor x) {
    }

    torch::Tensor AGLU::forward(torch::Tensor x) const {
        return xt::activations::aglu(x);
    }

}
