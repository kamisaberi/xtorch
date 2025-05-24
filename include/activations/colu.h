#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor colu(torch::Tensor x);

    struct CoLU : xt::Module {
    public:
        CoLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



