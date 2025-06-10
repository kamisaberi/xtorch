#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor phish(const torch::Tensor& x, double a = 1.0, double b = 1.0);

    struct Phish : xt::Module {
    public:
        Phish() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



