#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor crelu(torch::Tensor x);

    struct CReLU : xt::Module
    {
    public:
        CReLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
