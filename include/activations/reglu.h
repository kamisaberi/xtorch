#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor reglu(const torch::Tensor& x, int64_t dim = 1);

    struct ReGLU : xt::Module {
    public:
        ReGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



