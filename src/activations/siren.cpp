#include "include/activations/siren.h"

namespace xt::activations {
    torch::Tensor siren(torch::Tensor x) {
    }

    torch::Tensor Siren::forward(torch::Tensor x) const {
        return xt::activations::siren(x);
    }

}
