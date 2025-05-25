#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor recurrent_dropout(torch::Tensor x);

    struct RecurrentDropout : xt::Module {
    public:
        RecurrentDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



