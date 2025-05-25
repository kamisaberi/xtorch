#include "include/activations/serf.h"

namespace xt::activations {
    torch::Tensor serf(torch::Tensor x) {
    }

    torch::Tensor Serf::forward(torch::Tensor x) const {
        return xt::activations::serf(x);
    }

}
