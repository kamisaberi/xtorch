#pragma once

#include "common.h"

namespace xt::activations
{

    torch::Tensor shifted_softplus(const torch::Tensor& x, double shift_val = LN_2);

}
