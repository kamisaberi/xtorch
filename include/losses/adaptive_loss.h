#pragma once

#include "common.h"


namespace xt::losses {
    class AdaptiveLoss : xt::Module {
    public:
        AdaptiveLoss() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };

    torch::Tensor adaptive_loss(torch::Tensor x);
}
