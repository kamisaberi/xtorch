#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor auto_dropout(torch::Tensor x);

    struct AutoDropout : xt::Module {
    public:
        AutoDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



