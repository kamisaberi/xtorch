#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor poly(torch::Tensor x);

    struct Poly : xt::Module {
    public:
        Poly() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



