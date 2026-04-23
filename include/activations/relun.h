#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor relun(const torch::Tensor& x, double n_val = 1.0);

}



