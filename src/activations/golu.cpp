#include "include/activations/golu.h"

namespace xt::activations {
    torch::Tensor golu(torch::Tensor x) {
    }

    torch::Tensor GoLU::forward(torch::Tensor x) const {
        return xt::activations::golu(x);
    }

}
