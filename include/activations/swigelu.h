#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor swiglu(const torch::Tensor& x, int64_t dim = 1, double beta = 1.0);

}



