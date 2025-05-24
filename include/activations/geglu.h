#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor geglu(torch::Tensor x);

    struct GeGLU: xt::Module {
    public:
        GeGLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



