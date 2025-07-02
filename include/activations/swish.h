#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor swish(const torch::Tensor& x, double beta = 1.0);

    struct Swish : xt::Module
    {
    public:
        Swish() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
