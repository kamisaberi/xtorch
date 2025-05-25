#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor kaf(torch::Tensor x);

    struct KAF : xt::Module {
    public:
        KAF() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



