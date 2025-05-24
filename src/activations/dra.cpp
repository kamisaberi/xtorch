#include "include/activations/aglu.h"

namespace xt::activations {
    torch::Tensor dra(torch::Tensor x) {
    }

    torch::Tensor DRA::forward(torch::Tensor x) const {
        return xt::activations::dra(x);
    }

}
