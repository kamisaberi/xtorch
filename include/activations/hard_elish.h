#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hard_elish(torch::Tensor x);

    struct HardELiSH : xt::Module {
    public:
        HardELiSH() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



