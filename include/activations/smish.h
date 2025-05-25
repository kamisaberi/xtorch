#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smish(torch::Tensor x);

    struct Smish : xt::Module {
    public:
        Smish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



