#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mish(torch::Tensor x);

    struct Mish : xt::Module {
    };
}



