#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor tanh_exp(const torch::Tensor& x);

    struct TanhExp : xt::Module {
    public:

    private:
    };
}



