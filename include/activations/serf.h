#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor serf(torch::Tensor x);

    struct Serf : xt::Module {
    public:
        Serf() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



