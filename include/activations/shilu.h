#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor shilu(torch::Tensor x);

    struct ShiLU : xt::Module {
    public:
        ShiLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



