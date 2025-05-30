#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor attention_dropout(torch::Tensor x);

    struct AttentionDropout : xt::Module {
    public:
        AttentionDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



