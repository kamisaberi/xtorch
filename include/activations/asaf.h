#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor asaf(torch::Tensor x, double p_param = 1.0, double q_param = 1.0);

    struct ASAF : xt::Module {
    public:
        ASAF() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



