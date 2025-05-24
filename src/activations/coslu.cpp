#include "include/activations/coslu.h"

namespace xt::activations {
    torch::Tensor coslu(torch::Tensor x) {
    }

    torch::Tensor CosLU::forward(torch::Tensor x) const {
        return xt::activations::coslu(x);
    }

}
