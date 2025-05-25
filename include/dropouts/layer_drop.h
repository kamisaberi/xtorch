#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor layer_drop(torch::Tensor x);

    struct LayerDrop : xt::Module {
    public:
        LayerDrop() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



