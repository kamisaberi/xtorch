#include "include/activations/hard_elish.h"

namespace xt::activations {
    torch::Tensor hard_elish(torch::Tensor x) {
    }

    torch::Tensor HardELiSH::forward(torch::Tensor x) const {
        return xt::activations::hard_elish(x);
    }

}
