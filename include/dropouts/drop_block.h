#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_block(torch::Tensor x);

    struct DropBlock : xt::Module {
    public:
        DropBlock() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



