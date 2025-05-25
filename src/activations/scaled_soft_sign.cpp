#include "include/activations/scaled_soft_sign.h"

namespace xt::activations {
    torch::Tensor scaled_soft_sign(torch::Tensor x) {
    }

    torch::Tensor ScaledSoftSign::forward(torch::Tensor x) const {
        return xt::activations::scaled_soft_sign(x);
    }

}
