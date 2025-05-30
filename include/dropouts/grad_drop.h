#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor grad_drop(torch::Tensor x);

    struct GradDrop : xt::Module {
    public:
        GradDrop() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



