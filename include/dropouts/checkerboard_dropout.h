#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor checkerboard_dropout(torch::Tensor x);

    struct CheckerboardDropout : xt::Module {
    public:
        CheckerboardDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



