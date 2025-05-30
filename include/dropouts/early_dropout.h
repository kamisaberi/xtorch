#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor early_dropout(torch::Tensor x);

    struct EarlyDropout : xt::Module {
    public:
        EarlyDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



