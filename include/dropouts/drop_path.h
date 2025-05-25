#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_path(torch::Tensor x);

    struct DropPath : xt::Module {
    public:
        DropPath() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



