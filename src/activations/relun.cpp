#include "include/activations/relun.h"

namespace xt::activations {
    torch::Tensor relun(torch::Tensor x) {
    }

    torch::Tensor ReLUN::forward(torch::Tensor x) const {
        return xt::activations::relun(x);
    }

}
