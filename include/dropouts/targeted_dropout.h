#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor targeted_dropout(torch::Tensor x);

    struct TargetedDropout : xt::Module {
    public:
        TargetedDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



