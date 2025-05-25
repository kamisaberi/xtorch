#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor norm_lin_comb(torch::Tensor x);

    struct NormLinComb : xt::Module {
    public:
        NormLinComb() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



