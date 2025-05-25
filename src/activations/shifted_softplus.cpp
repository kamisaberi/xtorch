#include "include/activations/shifted_softplus.h"

namespace xt::activations {
    torch::Tensor shifted_softplus(torch::Tensor x) {
    }

    torch::Tensor ShiftedSoftplus::forward(torch::Tensor x) const {
        return xt::activations::shifted_softplus(x);
    }

}
