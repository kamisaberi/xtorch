#include "include/activations/nfn.h"

namespace xt::activations {
    torch::Tensor nfn(torch::Tensor x) {
    }

    torch::Tensor NFN::forward(torch::Tensor x) const {
        return xt::activations::nfn(x);
    }

}
