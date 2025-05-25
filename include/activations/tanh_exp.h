#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor tanh_exp(torch::Tensor x);

    struct TanhExp : xt::Module {
    public:
        TanhExp() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



