#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor poly(
        const torch::Tensor& x,
        const torch::Tensor& coefficients // Shape [degree, degree-1, ..., 1, 0]
    );


    struct Poly : xt::Module
    {
    public:

    private:
    };
}
