#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor adaptive_dropout(torch::Tensor x);

    struct AdaptiveDropout : xt::Module {
    public:
        AdaptiveDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



