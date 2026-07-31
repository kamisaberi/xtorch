#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor taaf(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Per-channel learnable parameter
        double beta = 0.0 // Global hyperparameter or fixed
    );


    struct TAAF : xt::Module
    {
    public:
        TAAF() = default;

    private:
    };
}
