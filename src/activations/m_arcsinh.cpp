#include "include/activations/m_arcsinh.h"

namespace xt::activations {
    torch::Tensor m_arcsinh(torch::Tensor x) {
    }

    torch::Tensor MArcsinh::forward(torch::Tensor x) const {
        return xt::activations::m_arcsinh(x);
    }

}
