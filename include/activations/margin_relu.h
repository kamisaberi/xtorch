#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor margin_relu(const torch::Tensor& x, double margin_neg = 0.1, double margin_pos = 0.9) ;

    struct MarginReLU : xt::Module {
    public:

    private:
    };
}



