#include "include/activations/asu.h"

namespace xt::activations {
    torch::Tensor asu(torch::Tensor x) {
        return  torch::zeros(10);
    }

    auto ASU::forward(std::initializer_list<std::any> tensors) -> std::any
    {

        return xt::activations::asu(torch::zeros(10));
    }

}
