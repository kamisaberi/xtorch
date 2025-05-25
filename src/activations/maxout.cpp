#include "include/activations/maxout.h"

namespace xt::activations {
    torch::Tensor maxout(torch::Tensor x) {
    }

    torch::Tensor Maxout::forward(torch::Tensor x) const {
        return xt::activations::maxout(x);
    }

}
