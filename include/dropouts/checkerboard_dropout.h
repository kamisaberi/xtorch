#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor checkerboard_dropout(torch::Tensor x);

    struct CheckerboardDropout : xt::Module {
    public:
        CheckerboardDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



