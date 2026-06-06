#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor squared_relu(const torch::Tensor& x);

    struct SquaredReLU : xt::Module {
    };
}



