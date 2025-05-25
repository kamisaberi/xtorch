#include "include/activations/serlu.h"

namespace xt::activations {
    torch::Tensor serlu(torch::Tensor x) {
    }

    torch::Tensor SERLU::forward(torch::Tensor x) const {
        return xt::activations::serlu(x);
    }

}
