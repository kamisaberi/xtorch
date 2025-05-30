#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_connect(torch::Tensor x);

    struct DropConnect : xt::Module {
    public:
        DropConnect() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



