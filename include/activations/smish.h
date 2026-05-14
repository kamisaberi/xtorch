#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smish(const torch::Tensor& x, double alpha = 1.0, double beta = 1.0);

    struct Smish : xt::Module {
    public:
        Smish() = default;


    private:
    };
}



