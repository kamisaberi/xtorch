#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor geglu(torch::Tensor x, int64_t dim = 1);

    struct GeGLU: xt::Module {
    public:
        GeGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



