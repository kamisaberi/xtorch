#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor taaf(torch::Tensor x);

    struct TAAF : xt::Module {
    public:
        TAAF() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



