#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor shifted_softplus(torch::Tensor x);

    struct ShiftedSoftplus : xt::Module {
    public:
        ShiftedSoftplus() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



