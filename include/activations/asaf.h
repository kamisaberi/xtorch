#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor asaf(torch::Tensor x);

    struct ASAF : xt::Module {
    public:
        ASAF() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



