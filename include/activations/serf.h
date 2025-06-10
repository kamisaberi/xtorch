#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor serf(const torch::Tensor& x, double k_param = 2.0, double lambda_param = 1.0);

    struct Serf : xt::Module {
    public:
        Serf() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



