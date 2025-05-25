#include "include/activations/nlsig.h"

namespace xt::activations {
    torch::Tensor nlsig(torch::Tensor x) {
    }

    torch::Tensor NLSIG::forward(torch::Tensor x) const {
        return xt::activations::nlsig(x);
    }

}
