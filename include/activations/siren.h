#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor siren(const torch::Tensor& x, double omega_0 = 30.0);

}



