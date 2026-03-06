#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor norm_lin_comb(
        const torch::Tensor& coefficients, // Shape (num_base_functions)
        double eps = 1e-5
    );


    struct NormLinComb : xt::Module
    {
    public:
        NormLinComb() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
