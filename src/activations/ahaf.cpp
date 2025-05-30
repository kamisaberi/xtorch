//TODO SHOULD IMPLEMENT
#include "include/activations/ahaf.h"

namespace xt::activations {
    torch::Tensor ahaf(torch::Tensor x) {
        return  torch::zeros(10);
    }

    auto AHAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::ahaf(torch::zeros(10));
    }

}
