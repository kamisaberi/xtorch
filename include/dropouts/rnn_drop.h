#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor rnn_drop(torch::Tensor x);

    struct RnnDrop : xt::Module {
    public:
        RnnDrop() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



