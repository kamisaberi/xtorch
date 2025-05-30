#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor kaf(torch::Tensor x);

    struct KAF : xt::Module {
    public:
        KAF() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



