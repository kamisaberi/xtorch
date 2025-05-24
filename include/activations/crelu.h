#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor crelu(torch::Tensor x);

    struct CReLU : xt::Module
    {
    public:
        CReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}
