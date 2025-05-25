#include "include/activations/leaf.h"

namespace xt::activations {
    torch::Tensor leaf(torch::Tensor x) {
    }

    torch::Tensor LEAF::forward(torch::Tensor x) const {
        return xt::activations::leaf(x);
    }

}
