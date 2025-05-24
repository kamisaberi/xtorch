#include "include/activations/aria.h"

namespace xt::activations {
    torch::Tensor aria(torch::Tensor x) {
    }

    torch::Tensor ARiA::forward(torch::Tensor x) const {
        return xt::activations::aria(x);
    }

}
