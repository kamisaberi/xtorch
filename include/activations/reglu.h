#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor reglu(torch::Tensor x);

    struct ReGLU : xt::Module {
    public:
        ReGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



