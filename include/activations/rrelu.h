#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor rrelu(torch::Tensor x);

    struct RReLU : xt::Module {
    public:
        RReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



