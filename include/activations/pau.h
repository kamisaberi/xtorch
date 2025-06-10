#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor pau(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, p_{m-1}, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, q_{n-1}, ..., q_1, 1.0] (q_0 is fixed to 1)
        double epsilon = 1e-7
    );


    struct PAU : xt::Module
    {
    public:
        PAU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
