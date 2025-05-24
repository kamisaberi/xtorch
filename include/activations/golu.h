#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor golu(torch::Tensor x);

    struct GoLU : xt::Module {
    public:
        GoLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



