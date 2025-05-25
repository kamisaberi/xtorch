#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hermite(torch::Tensor x);

    struct Hermite : xt::Module {
    public:
        Hermite() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



