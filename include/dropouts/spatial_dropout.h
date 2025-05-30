#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor spatial_dropout(torch::Tensor x);

    struct SpatialDropout : xt::Module {
    public:
        SpatialDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



