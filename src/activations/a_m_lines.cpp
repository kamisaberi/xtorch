#include "include/activations/a_m_lines.h"

namespace xt::activations {
    torch::Tensor am_lines(torch::Tensor x) {

    }

    auto AMLines::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::am_lines(torch::zeros(10));
    }

}
