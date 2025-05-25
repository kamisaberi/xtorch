#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor monte_carlo_dropout(torch::Tensor x);

    struct MonteCarloDropout : xt::Module {
    public:
        MonteCarloDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



