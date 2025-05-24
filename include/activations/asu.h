#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor asu(torch::Tensor x);

    struct ASU : xt::Module {
    public:
        ASU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



