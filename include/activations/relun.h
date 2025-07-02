#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor relun(const torch::Tensor& x, double n_val = 1.0);

    struct ReLUN : xt::Module {
    public:
        ReLUN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



