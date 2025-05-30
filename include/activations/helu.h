#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor helu(torch::Tensor x);

    struct HeLU : xt::Module {
    public:
        HeLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



