#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor kan(torch::Tensor x);

    struct KAN : xt::Module {
    public:
        KAN() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



