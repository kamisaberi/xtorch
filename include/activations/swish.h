#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor swish(torch::Tensor x);

    struct Swish : xt::Module
    {
    public:
        Swish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}
