#include "include/activations/tanh_exp.h"

namespace xt::activations {
    torch::Tensor tanh_exp(torch::Tensor x) {
    }

    torch::Tensor TanhExp::forward(torch::Tensor x) const {
        return xt::activations::tanh_exp(x);
    }

}
