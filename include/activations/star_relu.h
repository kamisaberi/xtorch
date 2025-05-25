#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor star_relu(torch::Tensor x);

    struct StarReLU: xt::Module {
    public:
        StarReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



