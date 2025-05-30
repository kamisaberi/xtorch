#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor evo_norms(torch::Tensor x);

    struct EvoNorms : xt::Module {
    public:
        EvoNorms() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



