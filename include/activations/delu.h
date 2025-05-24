#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor delu(torch::Tensor x);

    struct DELU : xt::Module
    {
    public:
        DELU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:


    };
}



