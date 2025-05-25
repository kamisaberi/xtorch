#include "include/activations/nipuna.h"

namespace xt::activations {
    torch::Tensor nipuna(torch::Tensor x) {
    }

    torch::Tensor Nipuna::forward(torch::Tensor x) const {
        return xt::activations::nipuna(x);
    }

}
