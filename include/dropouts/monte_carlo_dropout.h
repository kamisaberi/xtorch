#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor monte_carlo_dropout(torch::Tensor x);

    struct MonteCarloDropout : xt::Module {
    public:
        MonteCarloDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



