#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor pau(torch::Tensor x);

    struct PAU : xt::Module {
    public:
        PAU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



