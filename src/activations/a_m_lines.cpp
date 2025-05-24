#include "include/activations/a_m_lines.h"

namespace xt::activations {
    torch::Tensor am_lines(torch::Tensor x) {

    }

    torch::Tensor AMLines::forward(torch::Tensor x) const {
        return xt::activations::am_lines(x);
    }

}
