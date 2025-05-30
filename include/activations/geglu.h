#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor geglu(torch::Tensor x);

    struct GeGLU: xt::Module {
    public:
        GeGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



