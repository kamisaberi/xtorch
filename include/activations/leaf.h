#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor leaf(const torch::Tensor& x,
                       const torch::Tensor& s_weights, // Shape (L)
                       const torch::Tensor& r_weights, // Shape (L)
                       const torch::Tensor& u_weights // Shape (L)
    );

    struct LEAF : xt::Module {
    public:

    private:
    };
}



