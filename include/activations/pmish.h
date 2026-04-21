#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor pmish(const torch::Tensor& x, double alpha = 1.0, double beta = 0.5);

    struct PMish : xt::Module {
    public:

    private:
    };
}









