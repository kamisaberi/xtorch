#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gcu(torch::Tensor x);

    struct GCU : xt::Module {
    public:
        GCU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



