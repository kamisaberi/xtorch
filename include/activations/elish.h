#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor elish(torch::Tensor x);

    struct ELiSH : xt::Module {
    public:
        ELiSH() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



