#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor squared_relu(torch::Tensor x);

    struct SquaredReLU : xt::Module {
    public:
        SquaredReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



