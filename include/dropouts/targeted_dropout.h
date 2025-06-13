#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct TargetedDropout : xt::Module
    {
    public:
        TargetedDropout(double drop_fraction = 0.1, bool scale_kept = true);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double drop_fraction_; // Fraction of units to drop (those with smallest magnitudes)
        bool scale_kept_; // Whether to scale the kept units (like inverted dropout)
        double epsilon_ = 1e-7;
        void apply_targeted_dropout_to_slice(torch::Tensor slice);
    };
}
