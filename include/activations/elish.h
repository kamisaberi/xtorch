#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor elish(torch::Tensor x);

    struct ELiSH : xt::Module {
    public:
        ELiSH() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



