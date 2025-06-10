#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor splash(const torch::Tensor& x, double S = 1.0, double R = 0.5, double B = 1.0);

    struct SPLASH : xt::Module {
    public:
        SPLASH() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



