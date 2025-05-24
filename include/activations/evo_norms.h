#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor evo_norms(torch::Tensor x);

    struct EvoNorms : xt::Module {
    public:
        EvoNorms() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



