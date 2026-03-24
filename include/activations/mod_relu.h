#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mod_relu(const torch::Tensor& x, const torch::Tensor& b);

    struct ModReLU : xt::Module {
    };
}



