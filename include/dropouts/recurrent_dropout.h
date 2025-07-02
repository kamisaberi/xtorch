#pragma once

#include "common.h"

namespace xt::dropouts {

    struct RecurrentDropout : xt::Module {
    public:
        RecurrentDropout(double p_drop = 0.5);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_; // Probability of an element being zeroed out.
        double epsilon_ = 1e-7; // For numerical stability in division

    };
}



