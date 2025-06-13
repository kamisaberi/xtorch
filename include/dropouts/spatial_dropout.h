#pragma once

#include "common.h"

namespace xt::dropouts {

    struct SpatialDropout : xt::Module {
    public:
        SpatialDropout(double p_drop_channel = 0.5);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_channel_; // Probability of dropping an entire channel.
        double epsilon_ = 1e-7;   // For numerical stability

    };
}



