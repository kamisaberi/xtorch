#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor recurrent_dropout(torch::Tensor x);

    struct RecurrentDropout : xt::Module {
    public:
        RecurrentDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



