#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smooth_step(torch::Tensor x);

    struct SmoothStep : xt::Module {
    public:
        SmoothStep() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



