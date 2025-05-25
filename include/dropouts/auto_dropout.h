#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor auto_dropout(torch::Tensor x);

    struct AutoDropout : xt::Module {
    public:
        AutoDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



