#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor taaf(torch::Tensor x);

    struct TAAF : xt::Module {
    public:
        TAAF() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



