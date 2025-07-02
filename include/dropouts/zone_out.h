#pragma once

#include "common.h"

namespace xt::dropouts {

    struct ZoneOut : xt::Module {
    public:

        ZoneOut(double p_zoneout_h = 0.1);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_zoneout_h_; // Probability of a hidden unit's state being "zoned out" (copied from previous step)
        // This is often denoted as 'zh' in the paper for hidden state zoneout.
        // There can also be 'zc' for cell state zoneout in LSTMs. This module handles one state tensor.

    };
}



