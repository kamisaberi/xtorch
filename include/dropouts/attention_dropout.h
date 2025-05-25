#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor attention_dropout(torch::Tensor x);

    struct AttentionDropout : xt::Module {
    public:
        AttentionDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



