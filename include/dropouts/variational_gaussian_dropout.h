#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor variational_gaussian_dropout(torch::Tensor x);

    struct VariationalGaussianDropout : xt::Module {
    public:
        VariationalGaussianDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



