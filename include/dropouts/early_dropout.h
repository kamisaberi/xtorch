#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor early_dropout(torch::Tensor x);

    struct EarlyDropout : xt::Module {
    public:
        EarlyDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



