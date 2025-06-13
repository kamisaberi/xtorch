#pragma once

#include "common.h"

namespace xt::dropouts {

    struct GaussianDropout : xt::Module {
    public:
        GaussianDropout(double p_rate = 0.1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_rate_; // Corresponds to the 'p' in standard dropout, used to calculate alpha (variance)
        double alpha_; // Variance of the multiplicative noise (noise ~ N(0, alpha_)), so multiplier ~ N(1, alpha_)

    };
}



