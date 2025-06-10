#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor lin_comb(
        const torch::Tensor& x,
        const std::vector<std::function<torch::Tensor(const torch::Tensor&)>>& base_functions,
        const torch::Tensor& coefficients // Shape (num_base_functions)
    );

    struct LinComb : xt::Module {
    public:
        LinComb() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



