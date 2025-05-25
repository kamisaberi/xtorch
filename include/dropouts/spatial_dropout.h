#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor spatial_dropout(torch::Tensor x);

    struct SpatialDropout : xt::Module {
    public:
        SpatialDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



