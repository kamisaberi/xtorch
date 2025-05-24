#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor coslu(torch::Tensor x);

    struct CosLU : xt::Module {
    public:
        CosLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



