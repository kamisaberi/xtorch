#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor rrelu(torch::Tensor x);

    struct RReLU : xt::Module {
    public:
        RReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



