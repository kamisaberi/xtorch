#include "include/activations/reglu.h"

namespace xt::activations {
    torch::Tensor reglu(torch::Tensor x) {
    }

    torch::Tensor ReGLU::forward(torch::Tensor x) const {
        return xt::activations::reglu(x);
    }

}
