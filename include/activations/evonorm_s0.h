#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor evonorm_s0(torch::Tensor x);

    struct EvonormS0 : xt::Module {
    public:
        EvonormS0() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



