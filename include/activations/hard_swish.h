#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hard_swich(torch::Tensor x);

    struct HardSwish : xt::Module {
    public:
        HardSwish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



