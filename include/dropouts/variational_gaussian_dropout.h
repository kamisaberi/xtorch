#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor variational_gaussian_dropout(torch::Tensor x);

    struct VariationalGaussianDropout : xt::Module {
    public:
        VariationalGaussianDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



