#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor swiglu(torch::Tensor x);

    struct SwiGLU : xt::Module {
    public:
        SwiGLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



