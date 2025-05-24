#include "include/activations/asu.h"

namespace xt::activations {
    torch::Tensor asu(torch::Tensor x) {
    }

    torch::Tensor ASU::forward(torch::Tensor x) const {
        return xt::activations::asu(x);
    }

}
