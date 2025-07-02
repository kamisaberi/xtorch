#pragma once

#include "common.h"

namespace xt::dropouts {

    struct SpectralDropout : xt::Module {
    public:

        SpectralDropout(double p_drop_freq = 0.5);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_freq_; // Probability of dropping a frequency component (after considering Hermitian symmetry)
        double epsilon_ = 1e-7;

    };
}



