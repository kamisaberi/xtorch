#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smooth_step(const torch::Tensor& x, double edge0 = 0.0, double edge1 = 1.0);

    struct SmoothStep : xt::Module {
    public:
        SmoothStep() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



