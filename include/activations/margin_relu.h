#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor margin_relu(torch::Tensor x);

    struct MarginReLU : xt::Module {
    public:
        MarginReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



