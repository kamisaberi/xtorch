#include "include/activations/gclu.h"

namespace xt::activations {
    torch::Tensor gclu(torch::Tensor x) {
    }

    torch::Tensor GCLU::forward(torch::Tensor x) const {
        return xt::activations::gclu(x);
    }

}
