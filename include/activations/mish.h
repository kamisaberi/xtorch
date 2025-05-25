#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mish(torch::Tensor x);

    struct Mish : xt::Module {
    public:
        Mish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



