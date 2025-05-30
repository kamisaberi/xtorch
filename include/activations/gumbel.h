#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gumbel(torch::Tensor x);

    struct Gumbel : xt::Module {
    public:
        Gumbel() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



