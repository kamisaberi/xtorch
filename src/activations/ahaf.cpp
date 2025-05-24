#include "include/activations/ahaf.h"

namespace xt::activations {
    torch::Tensor ahaf(torch::Tensor x) {
    }

    torch::Tensor AHAF::forward(torch::Tensor x) const {
        return xt::activations::ahaf(x);
    }

}
