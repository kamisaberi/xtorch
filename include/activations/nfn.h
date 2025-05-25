#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nfn(torch::Tensor x);

    struct NFN : xt::Module {
    public:
        NFN() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



