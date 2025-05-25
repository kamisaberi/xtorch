#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor rnn_drop(torch::Tensor x);

    struct RnnDrop : xt::Module {
    public:
        RnnDrop() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



