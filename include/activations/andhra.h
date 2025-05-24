#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor andhra(torch::Tensor x);

    struct ANDHRA : xt::Module {
    public:
        ANDHRA() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



