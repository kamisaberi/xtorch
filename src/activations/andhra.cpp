#include "include/activations/andhra.h"

namespace xt::activations {
    torch::Tensor andhra(torch::Tensor x) {
    }

    torch::Tensor ANDHRA::forward(torch::Tensor x) const {
        return xt::activations::andhra(x);
    }

}
