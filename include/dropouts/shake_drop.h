#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor shake_dropout(torch::Tensor x);

    struct ShakeDropout : xt::Module {
    public:
        ShakeDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



