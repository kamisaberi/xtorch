#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor delu(torch::Tensor x);

    struct DELU : xt::Module
    {
    public:
        DELU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:


    };
}



