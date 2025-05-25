#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor targeted_dropout(torch::Tensor x);

    struct TargetedDropout : xt::Module {
    public:
        TargetedDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



