#include "include/activations/splash.h"

namespace xt::activations {
    torch::Tensor splash(torch::Tensor x) {
    }

    torch::Tensor SPLASH::forward(torch::Tensor x) const {
        return xt::activations::splash(x);
    }

}
