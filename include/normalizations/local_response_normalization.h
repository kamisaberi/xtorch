#pragma once

#include "common.h"

namespace xt::norm
{
    struct LocalResponseNorm : xt::Module
    {
    public:


        LocalResponseNorm(int64_t size, double alpha = 1e-4, double beta = 0.75, double k = 1.0);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t size_; // `n` in the formula: the number of channels to sum over for normalization.
        double alpha_; // Scaling parameter `alpha`.
        double beta_; // Exponent `beta`.
        double k_; // Additive constant `k`.


    };
}
