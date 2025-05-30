#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor swiglu(torch::Tensor x);

    struct SwiGLU : xt::Module {
    public:
        SwiGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



