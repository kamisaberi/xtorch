#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor grad_drop(torch::Tensor x);

    struct GradDrop : xt::Module {
    public:
        GradDrop() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



