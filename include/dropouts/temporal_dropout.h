#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor temporal_dropout(torch::Tensor x);

    struct TemporalDropout : xt::Module {
    public:
        TemporalDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



