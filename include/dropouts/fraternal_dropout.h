#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor fraternal_dropout(torch::Tensor x);

    struct FraternalDropout : xt::Module {
    public:
        FraternalDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



