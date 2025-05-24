#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gcu(torch::Tensor x);

    struct GCU : xt::Module {
    public:
        GCU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



