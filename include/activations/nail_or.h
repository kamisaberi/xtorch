#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nail_or(torch::Tensor x);

    struct NailOr : xt::Module {
    public:
        NailOr() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



