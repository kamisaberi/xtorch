#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor srelu(torch::Tensor x);

    struct SReLU : xt::Module {
    public:
        SReLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



