#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smooth_step(torch::Tensor x);

    struct SmoothStep : xt::Module {
    public:
        SmoothStep() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



