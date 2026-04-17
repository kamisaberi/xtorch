#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor reglu(const torch::Tensor& x, int64_t dim = 1);

}



