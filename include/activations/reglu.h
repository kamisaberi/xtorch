#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor reglu(torch::Tensor x);

    struct ReGLU : xt::Module {
    public:
        ReGLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



