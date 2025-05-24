#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor fem(torch::Tensor x);

    struct FEM : xt::Module {
    public:
        FEM() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



