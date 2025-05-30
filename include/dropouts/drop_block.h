#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_block(torch::Tensor x);

    struct DropBlock : xt::Module {
    public:
        DropBlock() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



