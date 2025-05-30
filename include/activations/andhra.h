#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor andhra(torch::Tensor x);

    struct ANDHRA : xt::Module {
    public:
        ANDHRA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



