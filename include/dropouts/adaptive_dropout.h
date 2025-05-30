#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor adaptive_dropout(torch::Tensor x);

    struct AdaptiveDropout : xt::Module {
    public:
        AdaptiveDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



