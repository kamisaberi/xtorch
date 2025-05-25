#include "include/activations/smooth_step.h"

namespace xt::activations {
    torch::Tensor smooth_step(torch::Tensor x) {
    }

    torch::Tensor SmoothStep::forward(torch::Tensor x) const {
        return xt::activations::smooth_step(x);
    }

}
