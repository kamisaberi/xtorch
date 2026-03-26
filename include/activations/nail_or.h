#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nail_or(const torch::Tensor& x, const torch::Tensor& z);

    struct NailOr : xt::Module {
    public:
    private:
    };
}



