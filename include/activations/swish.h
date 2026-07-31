#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor swish(const torch::Tensor& x, double beta = 1.0);

}
