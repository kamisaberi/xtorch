#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor scaled_soft_sign(torch::Tensor x);

    struct ScaledSoftSign : xt::Module {
    public:
        ScaledSoftSign() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



