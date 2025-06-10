#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor rational(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, ..., q_1] (q_0 is fixed to 1)
        double epsilon = 1e-7 // For denominator stability
    );


    struct Rational : xt::Module
    {
    public:
        Rational() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
