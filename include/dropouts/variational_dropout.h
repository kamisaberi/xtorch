#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor variational_dropout(torch::Tensor x);

    struct VariationalDropout : xt::Module {
    public:
        VariationalDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



