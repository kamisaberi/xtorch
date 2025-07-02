#pragma once

#include "common.h"

namespace xt::dropouts {

    struct DropPathway : xt::Module {
    public:
        DropPathway(double p_drop = 0.1);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_; // Probability of dropping the pathway (input tensor)
        double epsilon_ = 1e-7; // For numerical stability in division

    };
}



