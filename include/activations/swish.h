#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor swish(torch::Tensor x);

    struct Swish : xt::Module
    {
    public:
        Swish() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
