#pragma once

#include "common.h"

namespace xt::dropouts {

    struct FraternalDropout : xt::Module {
    public:
        FraternalDropout(double p_drop = 0.5, double p_same_mask = 0.5);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_; // Probability of an element being zeroed in the primary mask
        double p_same_mask_; // Probability that the beta network uses the same mask as alpha
                             // (1 - p_same_mask_) is the probability beta uses the inverse mask.
        double epsilon_ = 1e-7;

    };
}



