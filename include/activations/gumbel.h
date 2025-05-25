#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gumbel(torch::Tensor x);

    struct Gumbel : xt::Module {
    public:
        Gumbel() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



