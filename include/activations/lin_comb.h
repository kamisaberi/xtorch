#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor lin_comb(torch::Tensor x);

    struct LinComb : xt::Module {
    public:
        LinComb() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



