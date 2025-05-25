#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor temporal_dropout(torch::Tensor x);

    struct TemporalDropout : xt::Module {
    public:
        TemporalDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



