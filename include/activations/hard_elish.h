#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hard_elish(torch::Tensor x);

    struct HardELiSH : xt::Module {
    public:
        HardELiSH() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



