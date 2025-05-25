#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor leaf(torch::Tensor x);

    struct LEAF : xt::Module {
    public:
        LEAF() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



