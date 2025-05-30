#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor gaussian_dropout(torch::Tensor x);

    struct GaussianDropout : xt::Module {
    public:
        GaussianDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



