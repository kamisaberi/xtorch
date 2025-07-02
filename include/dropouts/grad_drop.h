#pragma once

#include "common.h"

namespace xt::dropouts {

    struct GradDrop : xt::Module {
    public:

        GradDrop(double p_drop_at_zero_activation = 0.5, double sensitivity = 1.0);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:


        double p_drop_at_zero_activation_; // Dropout probability if activation is zero
        double sensitivity_;               // How fast keep_prob increases with abs(activation)
        double base_logit_keep_at_zero_;   // Precomputed logit(1 - p_drop_at_zero_activation_)
        double epsilon_ = 1e-7;           // For numerical stability

    };
}



