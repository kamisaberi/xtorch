#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor gaussian_dropout(torch::Tensor x);

    struct GaussianDropout : xt::Module {
    public:
        GaussianDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



