#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor leaf(torch::Tensor x);

    struct LEAF : xt::Module {
    public:
        LEAF() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



