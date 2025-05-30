#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor e_swish(torch::Tensor x);

    struct ESwish : xt::Module {
    public:
        ESwish() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



