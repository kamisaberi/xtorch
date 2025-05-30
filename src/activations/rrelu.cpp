#include "include/activations/rrelu.h"

namespace xt::activations {
    torch::Tensor rrelu(torch::Tensor x) {
        return  torch::zeros(10);
    }
    auto RReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {


        return xt::activations::rrelu(torch::zeros(10));
    }

}
