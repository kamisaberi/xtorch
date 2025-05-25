#include "include/activations/swigelu.h"

namespace xt::activations {
    torch::Tensor swiglu(torch::Tensor x) {
    }

    torch::Tensor SwiGLU::forward(torch::Tensor x) const {
        return xt::activations::swiglu(x);
    }

}
