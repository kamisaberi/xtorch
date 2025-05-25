#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor relun(torch::Tensor x);

    struct ReLUN : xt::Module {
    public:
        ReLUN() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



