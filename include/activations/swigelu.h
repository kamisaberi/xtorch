#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor swiglu(const torch::Tensor& x, int64_t dim = 1, double beta = 1.0);

    struct SwiGLU : xt::Module {
    public:
        SwiGLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



