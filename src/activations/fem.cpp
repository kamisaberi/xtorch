#include "include/activations/fem.h"

namespace xt::activations {
    torch::Tensor fem(torch::Tensor x) {
    }

    torch::Tensor FEM::forward(torch::Tensor x) const {
        return xt::activations::fem(x);
    }

}
