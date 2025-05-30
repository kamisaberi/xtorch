#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_path(torch::Tensor x);

    struct DropPath : xt::Module {
    public:
        DropPath() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



