#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor aglu(torch::Tensor x);

    struct AGLU : xt::Module
    {
    public:
        AGLU() = default;
        torch::Tensor forward(torch::Tensor x) const override;
    private:
    };
}



