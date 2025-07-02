#pragma once

#include "common.h"

namespace xt::activations
{
    const double LN_2 = std::log(2.0); // More portable way to get log(2)

    torch::Tensor shifted_softplus(const torch::Tensor& x, double shift_val = LN_2);

    struct ShiftedSoftplus : xt::Module
    {
    public:
        ShiftedSoftplus() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
