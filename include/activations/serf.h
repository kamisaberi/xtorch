#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor serf(torch::Tensor x);

    struct Serf : xt::Module {
    public:
        Serf() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



