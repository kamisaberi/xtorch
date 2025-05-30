#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hard_swich(torch::Tensor x);

    struct HardSwish : xt::Module {
    public:
        HardSwish() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



