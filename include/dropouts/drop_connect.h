#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_connect(torch::Tensor x);

    struct DropConnect : xt::Module {
    public:
        DropConnect() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



