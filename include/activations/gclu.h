#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gclu(torch::Tensor x);

    struct GCLU : xt::Module {
    public:
        GCLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



