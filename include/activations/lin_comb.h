#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor lin_comb(torch::Tensor x);

    struct LinComb : xt::Module {
    public:
        LinComb() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



