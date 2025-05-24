#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor e_swish(torch::Tensor x);

    struct ESwish : xt::Module {
    public:
        ESwish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



