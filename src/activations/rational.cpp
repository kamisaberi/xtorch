#include "include/activations/rational.h"

namespace xt::activations {
    torch::Tensor rational(torch::Tensor x) {
    }

    torch::Tensor Rational::forward(torch::Tensor x) const {
        return xt::activations::rational(x);
    }

}
