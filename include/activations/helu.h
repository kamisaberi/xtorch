#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor helu(torch::Tensor x);

    struct HeLU : xt::Module {
    public:
        HeLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



