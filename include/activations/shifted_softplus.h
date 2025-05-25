#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor shifted_softplus(torch::Tensor x);

    struct ShiftedSoftplus : xt::Module {
    public:
        ShiftedSoftplus() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



