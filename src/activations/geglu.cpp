#include "include/activations/geglu.h"

namespace xt::activations {
    torch::Tensor geglu(torch::Tensor x) {
    }

    torch::Tensor GeGLU::forward(torch::Tensor x) const {
        return xt::activations::geglu(x);
    }

}
