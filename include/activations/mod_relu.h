#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mod_relu(torch::Tensor x);

    struct ModReLU : xt::Module {
    public:
        AGLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



