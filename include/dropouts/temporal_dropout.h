#pragma once

#include "common.h"

namespace xt::dropouts {

    struct TemporalDropout : xt::Module {
    public:
        TemporalDropout(double p_drop_timestep = 0.1, int time_dim = 1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_timestep_; // Probability of dropping an entire time step.
        int time_dim_;           // The dimension index representing time/sequence length.
        double epsilon_ = 1e-7;

    };
}



