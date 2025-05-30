#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor coslu(torch::Tensor x);

    struct CosLU : xt::Module {
    public:
        CosLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



