#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor shake_drop(torch::Tensor x);

    struct ShakeDrop : xt::Module {
    public:
        ShakeDrop() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



