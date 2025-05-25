#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor variational_dropout(torch::Tensor x);

    struct VariationalDropout : xt::Module {
    public:
        VariationalDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



