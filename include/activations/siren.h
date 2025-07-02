#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor siren(const torch::Tensor& x, double omega_0 = 30.0);

    struct Siren : xt::Module {
    public:
        Siren() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



