#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor layer_drop(torch::Tensor x);

    struct LayerDrop : xt::Module {
    public:
        LayerDrop() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



