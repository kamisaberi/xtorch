#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct ShakeDrop : xt::Module
    {
    public:
        ShakeDrop(double p_drop = 0.5);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_drop_; // Probability of applying alpha scaling (the "drop" or "shake" event)
        double alpha_range_min_ = -1.0;
        double alpha_range_max_ = 1.0;
        double beta_range_min_ = 0.0; // Corresponds to 1-c with c=1 from paper
        double beta_range_max_ = 2.0; // Corresponds to 1+c with c=1 from paper

        // For random number generation (per-instance, could be static for global seed)
        std::mt19937 gen_;
    };
}
