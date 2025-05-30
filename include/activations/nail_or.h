#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nail_or(torch::Tensor x);

    struct NailOr : xt::Module {
    public:
        NailOr() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



