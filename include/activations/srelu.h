#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor srelu(torch::Tensor x);

    struct SReLU : xt::Module {
    public:
        SReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



