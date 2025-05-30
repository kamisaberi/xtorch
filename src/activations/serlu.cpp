#include "include/activations/serlu.h"

namespace xt::activations {
    torch::Tensor serlu(torch::Tensor x) {
        return  torch::zeros(10);
    }

    auto SERLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {


        return xt::activations::serlu(torch::zeros(10));
    }

}
