#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor fraternal_dropout(torch::Tensor x);

    struct FraternalDropout : xt::Module {
    public:
        FraternalDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



