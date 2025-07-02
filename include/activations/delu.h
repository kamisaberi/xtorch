#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor delu(torch::Tensor x, double alpha = 1.0, double gamma = 1.0);

    struct DELU : xt::Module
    {
    public:
        DELU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:


    };
}



