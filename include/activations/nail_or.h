#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nail_or(const torch::Tensor& x, const torch::Tensor& z);

    struct NailOr : xt::Module {
    public:
        NailOr() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



