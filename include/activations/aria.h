#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor aria(torch::Tensor x);

    struct ARiA : xt::Module {
    public:
        ARiA() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



