#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor norm_lin_comb(torch::Tensor x);

    struct NormLinComb : xt::Module {
    public:
        NormLinComb() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



